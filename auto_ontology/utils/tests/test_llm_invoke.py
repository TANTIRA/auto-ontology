# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM invocation helpers."""

import contextlib
import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from auto_ontology.utils import llm_invoke
from auto_ontology.utils.llm_invoke import (
    _structured_output_kwargs,
    _system_messages_first,
)


@pytest.fixture(autouse=True)
def _isolate_fallback_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent client identities and cached fallback clients leaking across tests."""
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_IDS", set())
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_STRONG_REFS", {})
    monkeypatch.setattr(llm_invoke, "_REASONING_FALLBACK_CLIENT", None)


class _Model:
    """Stand-in exposing the model name the way langchain clients do."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def test_claude_uses_function_calling_without_a_tool_choice() -> None:
    """Bedrock behind LiteLLM rejects any explicit tool choice.

    The gateway maps tool_choice into ``toolConfig.toolChoice`` *and* forwards the
    original ``tool_choice.type``, and Bedrock refuses the pair:

        The additional field tool_choice/type conflicts with the existing field
        toolConfig.toolChoice.tool.

    Measured against the live endpoint: "auto", "any", "required" and langchain's
    default (naming the tool) all hit it; only omitting the choice succeeds.
    """
    kwargs = _structured_output_kwargs(_Model("aws/anthropic/bedrock-claude-opus-4-8"))

    assert kwargs == {"method": "function_calling", "tool_choice": None}


def test_claude_is_matched_on_either_vendor_or_family_name() -> None:
    for name in (
        "aws/anthropic/bedrock-claude-opus-4-8",
        "claude-3-5-sonnet",
        "ANTHROPIC/Claude",
    ):
        assert _structured_output_kwargs(_Model(name))["method"] == "function_calling"


def test_other_models_keep_the_langchain_default() -> None:
    """json_schema is preferred where supported, so nothing is forced for them."""
    for name in ("nvidia/nemotron-3.5-lightning-30b-a3b", "gpt-4o", ""):
        assert _structured_output_kwargs(_Model(name)) == {}


def test_a_client_exposing_model_instead_of_model_name_still_matches() -> None:
    """Some langchain clients expose `model` rather than `model_name`."""

    class _AltModel:
        model: Any = "aws/anthropic/bedrock-claude-opus-4-8"

    assert _structured_output_kwargs(_AltModel())["tool_choice"] is None


def _interleaved() -> list[BaseMessage]:
    """The shape sql_from_semantic sends: system, user, system, system, user."""
    return [
        SystemMessage(content="role"),
        HumanMessage(content="question"),
        SystemMessage(content="evidence"),
        SystemMessage(content="anchors"),
        HumanMessage(content="prompt"),
    ]


def test_claude_gets_its_system_messages_ahead_of_the_conversation() -> None:
    """azure/anthropic/* on the gateway 400s on a system message after a user one."""
    ordered = _system_messages_first(
        _Model("azure/anthropic/claude-opus-5-5"), _interleaved()
    )

    assert [m.content for m in ordered] == [
        "role",
        "evidence",
        "anchors",
        "question",
        "prompt",
    ]


def test_other_models_keep_their_message_order() -> None:
    messages = _interleaved()

    assert (
        _system_messages_first(
            _Model("nvidia/nemotron-3.5-lightning-30b-a3b"), messages
        )
        == messages
    )


class _SystemCheckModel(BaseModel):
    answer: str


def test_validation_retry_note_reaches_claude_ahead_of_the_conversation() -> None:
    """The retry appends a system message after the user turn; it must not stay there."""
    sent: list[list[BaseMessage]] = []

    class _Structured:
        def invoke(self, messages: list[BaseMessage]) -> _SystemCheckModel:
            sent.append(messages)
            if len(sent) == 1:
                _SystemCheckModel.model_validate({})
            return _SystemCheckModel(answer="ok")

    class _Claude(_Model):
        def with_structured_output(self, *_args: Any, **_kwargs: Any) -> _Structured:
            return _Structured()

    llm_invoke.safe_invoke_with_structured_output(
        _Claude("azure/anthropic/claude-opus-5-5"),  # type: ignore[arg-type]
        [SystemMessage(content="role"), HumanMessage(content="question")],
        _SystemCheckModel,
    )

    roles = [m.type for m in sent[1]]
    assert roles == ["system", "system", "human"]


def _bound() -> int | None:
    """Capacity of the bound currently in force, or None when unbounded."""
    guard = llm_invoke._INFLIGHT
    return None if guard is None else guard._initial_value  # type: ignore[attr-defined]


def test_limit_inflight_applies_and_releases_a_bound() -> None:
    assert _bound() is None
    with llm_invoke.limit_inflight(6):
        assert _bound() == 6
    assert _bound() is None


def test_overlapping_blocks_do_not_leak_a_bound() -> None:
    """Two blocks that overlap without nesting exit in non-stack order.

    A save/restore of one global has each block put back the value it captured,
    which drops the bound of a block still running and then leaves a stale bound
    installed with nothing active. Both are silent, and the second permanently
    re-imposes the ceiling this module exists to avoid.
    """
    a_inside = threading.Event()
    b_inside = threading.Event()
    seen_by_b: list[int | None] = []

    def first() -> None:
        with llm_invoke.limit_inflight(6):
            a_inside.set()
            b_inside.wait(5)  # leave only once B is also inside

    def second() -> None:
        a_inside.wait(5)
        with llm_invoke.limit_inflight(6):
            b_inside.set()
            time.sleep(0.2)  # outlive A
            seen_by_b.append(_bound())

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert seen_by_b == [6], "B lost its bound when A exited"
    assert _bound() is None, "a bound survived with no block active"


def test_the_most_restrictive_overlapping_bound_wins() -> None:
    with llm_invoke.limit_inflight(8):
        assert _bound() == 8
        with llm_invoke.limit_inflight(2):
            assert _bound() == 2
        # Leaving the tighter block must not leave the process unbounded.
        assert _bound() == 8
    assert _bound() is None


def test_a_non_positive_limit_leaves_another_blocks_bound_alone() -> None:
    """ "I need no bound" must not mean "remove someone else's"."""
    with llm_invoke.limit_inflight(4):
        with llm_invoke.limit_inflight(0):
            assert _bound() == 4
        assert _bound() == 4
    assert _bound() is None


def test_an_exception_inside_the_block_still_releases_the_bound() -> None:
    with contextlib.suppress(RuntimeError):
        with llm_invoke.limit_inflight(3):
            assert _bound() == 3
            raise RuntimeError("boom")
    assert _bound() is None


class _Schema:
    """Stand-in schema; the structured-invoke mock never validates it."""


class _Binding:
    """Stand-in for ``llm.bind()``, which wraps the client as ``bound``."""

    def __init__(self, bound: object) -> None:
        self.bound = bound


def test_non_reasoning_client_is_marked(monkeypatch: Any) -> None:
    sentinel = object()
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: sentinel)

    client = llm_invoke.get_non_reasoning_llm_client()

    assert client is sentinel
    assert llm_invoke._is_non_reasoning_client(client)
    assert llm_invoke._is_non_reasoning_client(_Binding(client))
    assert not llm_invoke._is_non_reasoning_client(object())
    assert not llm_invoke._is_non_reasoning_client(_Binding(object()))


def test_endless_bound_chain_is_not_followed_forever() -> None:
    """A ``MagicMock`` has a fresh ``bound`` on every access; the walk must stop."""
    assert not llm_invoke._is_non_reasoning_client(MagicMock())


def test_non_weakref_client_id_cannot_match_an_unrelated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = object()
    registered = object()
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_IDS", {id(candidate)})
    monkeypatch.setattr(
        llm_invoke,
        "_NON_REASONING_STRONG_REFS",
        {id(candidate): registered},
    )

    assert not llm_invoke._is_non_reasoning_client(candidate)


def test_structured_non_reasoning_call_falls_back_to_reasoning(
    monkeypatch: Any,
) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("429 Too Many Requests")
        return "from-reasoning"

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    result = llm_invoke.invoke_with_structured_output(primary, [], _Schema)

    assert result == "from-reasoning"
    assert calls == [primary, reasoning]


def test_bound_non_reasoning_client_falls_back(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    binding = _Binding(primary)
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> str:
        calls.append(llm)
        if llm is not reasoning:
            raise RuntimeError("down")
        return "ok"

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    result = llm_invoke.invoke_with_structured_output(binding, [], _Schema)

    assert result == "ok"
    assert calls == [binding, reasoning]


def test_reasoning_fallback_client_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = object()
    second = object()
    reasoning = object()
    builds: list[object] = []
    llm_invoke._remember_non_reasoning(first)
    llm_invoke._remember_non_reasoning(second)
    monkeypatch.setattr(
        llm_invoke,
        "get_llm_client",
        lambda **_kwargs: builds.append(reasoning) or reasoning,
    )

    assert llm_invoke._reasoning_fallback_client(first) is reasoning
    assert llm_invoke._reasoning_fallback_client(second) is reasoning
    assert builds == [reasoning]


def test_structured_fallback_returns_none_when_reasoning_also_fails(
    monkeypatch: Any,
) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> None:
        calls.append(llm)
        raise RuntimeError("429")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: primary)

    result = llm_invoke.safe_invoke_structured_nr([], _Schema)

    assert result is None
    assert calls == [primary, reasoning]


def test_structured_nr_uses_reasoning_when_non_reasoning_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = object()
    calls: list[object] = []

    monkeypatch.setattr(
        llm_invoke,
        "get_non_reasoning_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(EnvironmentError("missing key")),
    )
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(
        llm_invoke,
        "safe_invoke_with_structured_output",
        lambda llm, _messages, _schema: calls.append(llm) or "ok",
    )

    assert llm_invoke.safe_invoke_structured_nr([], _Schema) == "ok"
    assert calls == [reasoning]


def test_structured_nr_returns_none_when_neither_client_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_kwargs: object) -> object:
        raise EnvironmentError("missing key")

    monkeypatch.setattr(llm_invoke, "get_non_reasoning_llm_client", unavailable)
    monkeypatch.setattr(llm_invoke, "get_llm_client", unavailable)

    assert llm_invoke.safe_invoke_structured_nr([], _Schema) is None


def test_reasoning_structured_call_does_not_switch_models(monkeypatch: Any) -> None:
    reasoning = object()

    def invoke(_llm: object, _messages: object, _schema: object) -> None:
        raise RuntimeError("429")

    def fail_client(**_kwargs: object) -> object:
        raise AssertionError("reasoning call must not build a fallback client")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", fail_client)

    assert llm_invoke.invoke_with_structured_output(reasoning, [], _Schema) is None


def test_text_non_reasoning_call_falls_back_once(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("connection error")
        return "ok"

    monkeypatch.setattr(llm_invoke, "_safe_invoke_text", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    assert llm_invoke.safe_invoke_text(primary, "question") == "ok"
    assert calls == [primary, reasoning]


def test_text_nr_uses_reasoning_when_non_reasoning_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = object()
    calls: list[object] = []

    monkeypatch.setattr(
        llm_invoke,
        "get_non_reasoning_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(EnvironmentError("missing key")),
    )
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(
        llm_invoke,
        "_safe_invoke_text",
        lambda llm, _prompt: calls.append(llm) or "ok",
    )

    assert llm_invoke.safe_invoke_text_nr("question") == "ok"
    assert calls == [reasoning]


def test_text_fallback_is_not_attempted_twice(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        raise RuntimeError("429")

    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: primary)
    monkeypatch.setattr(llm_invoke, "_safe_invoke_text", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    try:
        llm_invoke.safe_invoke_text_nr("question")
    except RuntimeError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("both models failing must raise")

    assert calls == [primary, reasoning]


def test_invoke_text_falls_back_for_a_non_reasoning_client(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("down")
        return "pql"

    monkeypatch.setattr(llm_invoke, "_invoke_text_once", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    assert llm_invoke.invoke_text(primary, "predict") == "pql"
    assert calls == [primary, reasoning]


def test_missing_reasoning_client_keeps_the_original_failure(monkeypatch: Any) -> None:
    primary = object()
    llm_invoke._remember_non_reasoning(primary)

    def invoke(_llm: object, _messages: object, _schema: object) -> None:
        raise RuntimeError("429")

    def unavailable(**_kwargs: object) -> object:
        raise EnvironmentError("REASONING_API_KEY is not set")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", unavailable)

    assert llm_invoke.invoke_with_structured_output(primary, [], _Schema) is None


def _anthropic_client(
    monkeypatch: pytest.MonkeyPatch, model: str, response: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """A real ChatAnthropic from ``_build_client`` answering from a fake transport."""
    monkeypatch.delenv("ANTHROPIC_API_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx2.Response(200, json=response)

    llm = llm_invoke._build_client(
        model=model,
        api_key="sk-ant-test",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
        max_tokens=512,
    )
    llm.__dict__["_client"] = anthropic.Anthropic(
        api_key="sk-ant-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    return llm, requests


def _message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class _Answer(BaseModel):
    value: str


_TOOL_CALL = {
    "type": "tool_use",
    "id": "t1",
    "name": "_Answer",
    "input": {"value": "x"},
}


@pytest.mark.filterwarnings("ignore:You are attempting to use structured output")
def test_anthropic_model_calls_anthropic_without_rejected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key stays off the NVIDIA endpoint and Opus 5.5's 400s are not triggered."""
    llm, requests = _anthropic_client(
        monkeypatch,
        "anthropic/claude-opus-5-5",
        _message([_TOOL_CALL], "tool_use"),
    )

    result = llm_invoke.safe_invoke_with_structured_output(
        llm,
        [
            SystemMessage(content="s"),
            HumanMessage(content="q"),
            SystemMessage(content="r"),
        ],
        _Answer,
    )

    assert result == _Answer(value="x")
    assert requests[0]["url"].startswith("https://api.anthropic.com/")
    body = requests[0]["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert "temperature" not in body
    assert "tool_choice" not in body


def test_anthropic_model_without_server_fallback_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, requests = _anthropic_client(
        monkeypatch,
        "anthropic/claude-haiku-5-5",
        _message([_TOOL_CALL], "tool_use"),
    )

    llm_invoke.safe_invoke_with_structured_output(
        llm, [HumanMessage(content="q")], _Answer
    )

    assert "fallbacks" not in requests[0]["body"]


@pytest.mark.filterwarnings("ignore:You are attempting to use structured output")
def test_anthropic_answer_without_tool_call_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_invoke.time, "sleep", lambda _: None)
    llm, requests = _anthropic_client(
        monkeypatch,
        "anthropic/claude-opus-5-5",
        _message([{"type": "text", "text": "no tool"}], "end_turn"),
    )

    with pytest.raises(OutputParserException):
        llm_invoke.safe_invoke_with_structured_output(
            llm, [HumanMessage(content="q")], _Answer
        )

    assert len(requests) == llm_invoke.RETRY_MAX_ATTEMPTS


def test_anthropic_text_response_drops_thinking_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, _ = _anthropic_client(
        monkeypatch,
        "anthropic/claude-opus-5-5",
        _message(
            [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "hello"},
            ],
            "end_turn",
        ),
    )

    assert llm_invoke.safe_invoke_text(llm, "q") == "hello"
    assert llm_invoke.invoke_text(llm, "q") == "hello"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (404, False), (429, True), (500, True), (529, True)],
)
def test_anthropic_status_codes_classify_for_retry(
    monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool
) -> None:
    """ChatAnthropic wraps SDK errors in subclasses; the base class must still match."""
    llm, _ = _anthropic_client(monkeypatch, "anthropic/claude-opus-5-5", {})
    llm.__dict__["_client"] = anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(
                    status,
                    json={"type": "error", "error": {"type": "x", "message": "m"}},
                )
            )
        ),
    )

    with pytest.raises(anthropic.APIStatusError) as caught:
        llm.invoke("q")

    assert llm_invoke._is_retryable(caught.value) is retryable
