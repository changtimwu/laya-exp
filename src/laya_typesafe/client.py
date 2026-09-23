"""TypeSafeClient / AsyncTypeSafeClient backed by local Laya-MLX inference."""

import asyncio
import json
import os
import threading
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ValidationError

from .types import (
    Choice,
    ListModelsResponse,
    ModelMetadata,
    Noul,
    Question,
    Score,
    SystemOneResponse,
)

# Jev's documented request limits (docs.typesafe.ai/api). Laya itself enforces neither.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10

DEFAULT_MODEL = "aac6fef/laya-mlx"
MODEL_ALIASES = {
    "jev-latest": DEFAULT_MODEL,
    "laya": DEFAULT_MODEL,
    "laya-multilingual": "aac6fef/laya-multilingual-mlx",
    "laya-typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}
MODEL_CATALOG = (
    ModelMetadata(name="laya", description="Laya 421M, ModernBERT-large, English, 512-token context",
                  release_date="2026-01-01"),
    ModelMetadata(name="laya-multilingual", description="Laya 322M, mmBERT-base, multilingual, 1024-token context",
                  release_date="2026-01-01"),
    ModelMetadata(name="laya-typed-decisions", description="Laya 421M, ModernBERT-large, 1024-token context",
                  release_date="2026-01-01"),
)


class TypeSafeError(Exception):
    """Invalid client usage, raised before any inference runs."""


class TypeSafeAPIError(TypeSafeError):
    """The equivalent of an unsuccessful API response (422 for an invalid request)."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class TypeSafeAPIConnectionError(TypeSafeError):
    pass


class TypeSafeAPIResponseValidationError(TypeSafeError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    """Accepted for signature compatibility; local inference never retries."""

    max_retries: int = 2


# Loaded agents are shared across clients: a checkpoint is ~1 GiB and loading takes seconds.
_AGENTS: dict[tuple, tuple[Any, threading.Lock]] = {}
_AGENTS_LOCK = threading.Lock()


def _get_agent(model_id: str, options: Mapping[str, Any]):
    key = (model_id, tuple(sorted(options.items())))
    with _AGENTS_LOCK:
        if key not in _AGENTS:
            import laya_mlx

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # temperature clamp notice
                _AGENTS[key] = (laya_mlx.load(model_id, **options), threading.Lock())
        return _AGENTS[key]


def _text(value) -> str:
    """Instructions as model text. Structured values keep non-ASCII characters readable."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _to_laya(qid: str, question: Question) -> dict:
    if isinstance(question, BaseModel):
        q = question.model_dump()
    elif isinstance(question, Mapping):
        kind = question.get("type")
        cls = {"noul": Noul, "choice": Choice, "score": Score}.get(kind)
        if cls is None:
            raise TypeSafeAPIError(f"questions.{qid}.type: expected 'noul', 'choice' or 'score', got {kind!r}")
        try:
            q = cls.model_validate(dict(question)).model_dump()
        except ValidationError as e:
            raise TypeSafeAPIError(f"questions.{qid}: {e}") from None
    else:
        raise TypeSafeError(f"questions.{qid}: expected a Noul, Choice, Score or question dict")

    kind, criteria = q["type"], q.get("criteria")
    if kind == "choice":
        if not criteria:
            raise TypeSafeAPIError(f"questions.{qid}.criteria: a choice needs at least one option")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise TypeSafeAPIError(f"questions.{qid}.criteria: at most {MAX_CHOICE_OPTIONS} options")
        criteria = dict(criteria)
    elif kind == "score":
        if not criteria:
            raise TypeSafeError(f"questions.{qid}.criteria: a score needs a nonempty list of levels")
        if len(criteria) > MAX_SCORE_LEVELS:
            raise TypeSafeAPIError(f"questions.{qid}.criteria: at most {MAX_SCORE_LEVELS} levels")
        criteria = list(criteria)
    return {"type": kind, "instructions": _text(q.get("instructions")), "criteria": criteria}


class _Base:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout=None,
        headers: Mapping[str, str] | None = None,
        transport=None,
        http_client=None,
        base_url: str | None = None,
        laya_options: Mapping[str, Any] | None = None,
    ):
        # api_key, retry, timeout, headers, transport, http_client and base_url are accepted so
        # existing Jev code constructs unchanged; local inference has no use for them.
        if transport is not None and http_client is not None:
            raise ValueError("transport and http_client are mutually exclusive")
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or "jev-latest"
        self.laya_options = dict(laya_options or {})
        self.models = _Models()

    @staticmethod
    def resolve_model(name: str) -> str:
        return MODEL_ALIASES.get(name, name)

    def _system_one(self, state, questions, *, model=None, extra_body=None, response_model=None):
        if state is None:
            raise TypeSafeError("state cannot be None")
        if not questions:
            raise TypeSafeError("questions must be a nonempty mapping")
        if extra_body:
            state = extra_body.get("state", state)
            questions = extra_body.get("questions", questions)
            model = extra_body.get("model", model)
        laya_questions = {qid: _to_laya(qid, q) for qid, q in questions.items()}

        model_id = self.resolve_model(model or self.model)
        agent, lock = _get_agent(model_id, self.laya_options)
        try:
            with lock:
                raw = agent.predict(state, laya_questions)
        except ValueError as e:  # e.g. options overflow Laya's head token budget
            raise TypeSafeAPIError(str(e)) from None

        body = {
            "model": f"laya:{model_id}",
            "answers": raw["answers"],
            "usage": raw["usage"],
            "request_id": uuid.uuid4().hex,
        }
        try:
            return (response_model or SystemOneResponse).model_validate(body)
        except ValidationError as e:
            raise TypeSafeAPIResponseValidationError(str(e)) from None


class _Models:
    def list(self, *, retry=None, timeout=None, extra_headers=None) -> ListModelsResponse:
        return ListModelsResponse(models=MODEL_CATALOG, request_id=uuid.uuid4().hex)


class TypeSafeClient(_Base):
    """Drop-in for `typesafe_sdk.TypeSafeClient` that answers with a local Laya checkpoint.

    `model` accepts "jev-latest" (mapped to the English Laya checkpoint), the aliases in
    MODEL_ALIASES, or any Laya Hub id / local path. `laya_options` is forwarded to
    `laya_mlx.load` (dtype, batch_size, compile, cache_prompts, ...).
    """

    def system_one(self, state, questions, *, model=None, retry=None, timeout=None,
                   extra_headers=None, extra_body=None, response_model=None):
        return self._system_one(state, questions, model=model, extra_body=extra_body,
                                response_model=response_model)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncTypeSafeClient(_Base):
    """Async variant; inference runs in a worker thread, serialized per checkpoint."""

    async def system_one(self, state, questions, *, model=None, retry=None, timeout=None,
                         extra_headers=None, extra_body=None, response_model=None):
        return await asyncio.to_thread(self._system_one, state, questions, model=model,
                                       extra_body=extra_body, response_model=response_model)

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
