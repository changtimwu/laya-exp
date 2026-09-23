"""Question and answer types mirroring `typesafe_sdk` (docs.typesafe.ai/sdk/python/api)."""

from functools import cached_property
from typing import Annotated, Any, Literal, Mapping, NotRequired, Sequence, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict, Field

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONContent = str | dict[str, JSONValue] | list[JSONValue]


# ---------------------------------------------------------------- questions


class NoulCriteria(TypedDict, total=False):
    true: JSONContent | None
    false: JSONContent | None


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Noul(_Question):
    """A yes/no question with optional descriptions for either outcome."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class Choice(_Question):
    """A question that selects between named alternatives."""

    type: Literal["choice"] = "choice"
    criteria: Mapping[str, JSONContent | None]
    instructions: JSONContent | None = None


class Score(_Question):
    """A question that assigns a score using an ordered rubric."""

    type: Literal["score"] = "score"
    criteria: Sequence[JSONContent]
    instructions: JSONContent | None = None


class NoulModel(TypedDict):
    type: Literal["noul"]
    instructions: NotRequired[JSONContent | None]
    criteria: NotRequired[NoulCriteria | None]


class ChoiceModel(TypedDict):
    type: Literal["choice"]
    instructions: NotRequired[JSONContent | None]
    criteria: Mapping[str, JSONContent | None]


class ScoreModel(TypedDict):
    type: Literal["score"]
    instructions: NotRequired[JSONContent | None]
    criteria: Sequence[JSONContent]


QuestionModel: TypeAlias = NoulModel | ChoiceModel | ScoreModel
Question: TypeAlias = Noul | Choice | Score | QuestionModel
Questions: TypeAlias = Mapping[str, Question]


# ---------------------------------------------------------------- answers


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class NoulAnswer(_Answer):
    """A yes/no answer. Like Jev, carries no `confidence`."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(_Answer):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(_Answer):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, str | dict[str, Any] | list[Any]]
    probabilities: dict[int, float]


Answer: TypeAlias = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(_Answer):
    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(_Answer):
    """Answers keyed by question name, with model and usage metadata."""

    model_config = ConfigDict(extra="ignore", frozen=True, ignored_types=(cached_property,))

    model: str
    usage: Usage
    answers: dict[str, Answer]
    request_id: str = ""

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}

    @property
    def raw_http_response(self):
        raise AttributeError("laya_typesafe runs locally; there is no HTTP response")


class ModelMetadata(_Answer):
    name: str
    description: str
    release_date: str


class ListModelsResponse(_Answer):
    models: tuple[ModelMetadata, ...]
    request_id: str = ""
