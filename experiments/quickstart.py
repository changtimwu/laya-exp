"""Minimal Laya typed-decision experiment. Run: uv run experiments/quickstart.py"""
import json
import time

import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx", dtype="float16")

state = "I was billed twice. Please refund the duplicate today."
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
            "sales": "new purchases",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
}

agent.predict(state, questions)  # warm-up
t = time.perf_counter()
result = agent.predict(state, questions)
print(json.dumps(result["answers"], indent=2, ensure_ascii=False))
print(f"latency: {(time.perf_counter() - t) * 1000:.1f} ms")
