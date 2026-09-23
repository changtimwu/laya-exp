import asyncio
import sys

import pytest
from pydantic import BaseModel

import laya_typesafe as ts
from laya_typesafe.client import _to_laya

TICKET = "My running shoes arrived in the wrong size. Can I swap them for a size 10?"
DEPARTMENT = ts.Choice(
    instructions="Which team should handle this?",
    criteria={
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    },
)


# ------------------------------------------------ request validation (no model load)


def test_empty_questions_rejected():
    with pytest.raises(ts.TypeSafeError):
        ts.TypeSafeClient().system_one(state="hi", questions={})


def test_none_state_rejected():
    with pytest.raises(ts.TypeSafeError):
        ts.TypeSafeClient().system_one(state=None, questions={"q": ts.Noul(instructions="?")})


def test_empty_score_criteria_rejected():
    with pytest.raises(ts.TypeSafeError):
        _to_laya("s", ts.Score(instructions="?", criteria=[]))


def test_jev_limits_enforced():
    with pytest.raises(ts.TypeSafeAPIError) as e:
        _to_laya("c", {"type": "choice", "criteria": {str(i): None for i in range(256)}})
    assert e.value.status_code == 422
    with pytest.raises(ts.TypeSafeAPIError):
        _to_laya("s", ts.Score(criteria=[str(i) for i in range(11)]))


def test_unknown_type_and_extra_fields_rejected():
    with pytest.raises(ts.TypeSafeAPIError):
        _to_laya("q", {"type": "rank", "criteria": ["a"]})
    with pytest.raises(ts.TypeSafeAPIError):
        _to_laya("q", {"type": "noul", "instructions": "?", "bogus": 1})


def test_structured_instructions_keep_non_ascii():
    q = _to_laya("q", ts.Noul(instructions={"question": "客戶要求退款嗎？"}))
    assert "退款" in q["instructions"]


def test_dict_and_object_questions_convert_identically():
    as_dict = {"type": "choice", "instructions": DEPARTMENT.instructions, "criteria": dict(DEPARTMENT.criteria)}
    assert _to_laya("a", as_dict) == _to_laya("b", DEPARTMENT)


def test_missing_instructions_allowed():
    assert _to_laya("q", {"type": "noul"})["instructions"] == ""


def test_install_registers_module():
    saved = sys.modules.pop("typesafe_sdk", None)
    try:
        ts.install()
        from typesafe_sdk import TypeSafeClient  # noqa: F401

        assert sys.modules["typesafe_sdk"] is ts
    finally:
        sys.modules.pop("typesafe_sdk", None)
        if saved is not None:
            sys.modules["typesafe_sdk"] = saved


def test_models_list():
    names = [m.name for m in ts.TypeSafeClient().models.list().models]
    assert names == ["laya", "laya-multilingual", "laya-typed-decisions"]


# ------------------------------------------------ inference (loads the English checkpoint)


@pytest.fixture(scope="module")
def client():
    with ts.TypeSafeClient() as c:
        yield c


def test_response_shape_matches_sdk(client):
    r = client.system_one(
        state=TICKET,
        questions={
            "department": DEPARTMENT,
            "urgent": ts.Noul(instructions="Does the customer say this is urgent?"),
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Concerned but civil", "Very angry"]},
        },
    )
    assert isinstance(r, ts.SystemOneResponse)
    assert r.model == "laya:aac6fef/laya-mlx"
    assert r.usage.output_tokens == 0 and r.usage.input_tokens > 0
    assert r.request_id

    dep = r.answers["department"]
    assert isinstance(dep, ts.ChoiceAnswer) and dep.choice == "returns"
    assert set(dep.probabilities) == {"returns", "shipping", "billing"}
    assert abs(sum(dep.probabilities.values()) - 1) < 1e-2

    urgent = r.answers["urgent"]
    assert isinstance(urgent, ts.NoulAnswer) and 0 <= urgent.noul <= 1
    assert not hasattr(urgent, "confidence")  # Jev nouls carry no confidence

    fr = r.answers["frustration"]
    assert isinstance(fr, ts.ScoreAnswer)
    assert set(fr.probabilities) == {0, 1, 2} and fr.legend[2] == "Very angry"
    assert 0 <= fr.score <= 2

    assert set(r.nouls) == {"urgent"} and set(r.choices) == {"department"} and set(r.scores) == {"frustration"}


def test_structured_criteria_and_state(client):
    r = client.system_one(
        state={"message": "Reply with your login password so we can release your bonus."},
        questions={"cred": ts.Noul(
            instructions={"question": "Does `message` ask the recipient to disclose a credential?"},
            criteria={"true": {"what": "Asks for a password, PIN or code"}, "false": "No credential requested"},
        )},
    )
    assert r.answers["cred"].noul > 0.5


def test_custom_response_model(client):
    class Minimal(BaseModel):
        model: str
        answers: dict

    r = client.system_one(state=TICKET, questions={"d": DEPARTMENT}, response_model=Minimal)
    assert isinstance(r, Minimal) and r.answers["d"]["choice"] == "returns"


def test_async_client():
    async def main():
        async with ts.AsyncTypeSafeClient() as c:
            return await asyncio.gather(*(c.system_one(state=TICKET, questions={"d": DEPARTMENT}) for _ in range(3)))

    assert all(r.answers["d"].choice == "returns" for r in asyncio.run(main()))
