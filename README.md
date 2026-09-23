# laya-exp

Experiments with [laya-mlx](https://github.com/mizorewww/laya-mlx), an open-weight MLX port of Laya typed decisions, compared against TypeSafe's Jev ([docs](https://docs.typesafe.ai/primitives)).

```bash
uv sync
uv run experiments/quickstart.py     # minimal choice/score/noul call
uv run experiments/jev_parity.py     # replay Jev doc examples on all three checkpoints
uv run experiments/limits.py         # option capacity, >10 levels, state truncation
uv run experiments/jev_sdk_example.py # Jev docs code, unmodified, on Laya
uv run pytest
```

Findings: [#1 Laya vs Jev](https://github.com/changtimwu/laya-exp/issues/1).

## `laya_typesafe`: run Jev code on Laya

A local drop-in for [`typesafe_sdk`](https://docs.typesafe.ai/sdk/python): same `Noul` / `Choice` / `Score`
question objects (or dicts), `TypeSafeClient` / `AsyncTypeSafeClient.system_one(...)`, and
`SystemOneResponse` with typed `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer`, `.nouls` / `.choices` / `.scores`.

```python
import laya_typesafe; laya_typesafe.install()   # makes `import typesafe_sdk` resolve here
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient() as client:                  # model="jev-latest" -> English Laya checkpoint
    r = client.system_one(state="I was charged twice.", questions={
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
    })
print(r.answers["billing"].noul, r.answers["tone"].choice)
```

- `model`: `jev-latest` / `laya`, `laya-multilingual`, `laya-typed-decisions`, or any Laya Hub id / path.
  `laya_options={...}` is forwarded to `laya_mlx.load` (dtype, compile, cache_prompts, ...).
- Enforces Jev's request limits (255 options, 10 levels) and raises `TypeSafeAPIError` (422) like the API would,
  including when options overflow Laya's token budget.
- Normalizes Laya's output to Jev's schema: no `confidence` on nouls, no `action` field, int-keyed score maps.
  Structured instructions keep non-ASCII text (works around laya-mlx's `ensure_ascii` escaping).
- Not emulated: network options (`api_key`, `retry`, `timeout`, `base_url`, ...) are accepted and ignored;
  `raw_http_response` is unavailable. **Answer values are Laya's** — `confidence` uses Laya's entropy formula
  and thresholds tuned for Jev will not transfer (see #1).

`upstream/` is a local clone of laya-mlx for reference (git-ignored).
