# laya-exp

Experiments with [laya-mlx](https://github.com/mizorewww/laya-mlx), an open-weight MLX port of Laya typed decisions, compared against TypeSafe's Jev ([docs](https://docs.typesafe.ai/primitives)).

```bash
uv sync
uv run experiments/quickstart.py     # minimal choice/score/noul call
uv run experiments/jev_parity.py     # replay Jev doc examples on all three checkpoints
uv run experiments/limits.py         # option capacity, >10 levels, state truncation
```

`upstream/` is a local clone of laya-mlx for reference (git-ignored).
