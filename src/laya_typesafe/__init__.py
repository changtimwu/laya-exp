"""A `typesafe_sdk`-compatible client that runs Jev-style System One requests on local Laya-MLX.

    from laya_typesafe import Choice, Noul, Score, TypeSafeClient

or, to run unmodified code that does `from typesafe_sdk import ...`:

    import laya_typesafe; laya_typesafe.install()
"""

import sys

from .client import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MODEL_ALIASES,
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeClient,
    TypeSafeError,
)
from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    ChoiceModel,
    JSONContent,
    JSONValue,
    ListModelsResponse,
    ModelMetadata,
    Noul,
    NoulAnswer,
    NoulCriteria,
    NoulModel,
    Question,
    QuestionModel,
    Questions,
    Score,
    ScoreAnswer,
    ScoreModel,
    SystemOneResponse,
    Usage,
)


def install(force: bool = False) -> None:
    """Register this module as `typesafe_sdk`, so existing Jev code imports it unchanged.

    Does nothing if the real SDK is already imported, unless `force=True`.
    """
    if force or "typesafe_sdk" not in sys.modules:
        sys.modules["typesafe_sdk"] = sys.modules[__name__]


__all__ = [
    "MAX_CHOICE_OPTIONS", "MAX_SCORE_LEVELS", "MODEL_ALIASES",
    "Answer", "AsyncTypeSafeClient", "Choice", "ChoiceAnswer", "ChoiceModel", "JSONContent",
    "JSONValue", "ListModelsResponse", "ModelMetadata", "Noul", "NoulAnswer", "NoulCriteria",
    "NoulModel", "Question", "QuestionModel", "Questions", "RetryPolicy", "Score", "ScoreAnswer",
    "ScoreModel", "SystemOneResponse", "TypeSafeAPIConnectionError", "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError", "TypeSafeClient", "TypeSafeError", "Usage", "install",
]
