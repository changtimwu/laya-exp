"""Replay examples from Jev's docs (docs.typesafe.ai/primitives) on Laya and compare answers.

Jev reference values are copied from the published jev-1.13.0 responses in those docs.
Run: uv run experiments/jev_parity.py [model_id ...]
"""
import sys
import warnings

import laya_mlx as laya

warnings.filterwarnings("ignore", category=RuntimeWarning)

SEVERITY = {
    "type": "score",
    "instructions": "How severe is the reported issue?",
    "criteria": [
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but workaround exists",
        "Blocking issue; no workaround exists",
    ],
}
DEPARTMENT = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    },
}
ESCALATION = {"type": "noul", "instructions": "Is the customer asking for a human agent?"}

# (name, state, questions, {qid: (field, jev_value)})
CASES = [
    ("choice/basic",
     "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
     {"department": DEPARTMENT}, {"department": ("choice", "returns")}),
    ("choice/5q-ambiguous",
     "Shoes arrived two weeks late and in the wrong size. Also I see two charges of $120 on my card. What are you going to do about this?",
     {
         "department": DEPARTMENT,
         "return_reason": {"type": "choice", "instructions": "If the customer wants to return something, why?", "criteria": {
             "wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered",
             "damaged": "The item arrived broken or faulty",
             "changed_mind": "The item is fine, the customer no longer wants it",
             "other": "A return reason that fits none of the above"}},
         "shipping_issue": {"type": "choice", "instructions": "If this is a shipping problem, which kind is it?", "criteria": {
             "not_delivered": "The package never arrived", "delayed": "The package is late but still on its way",
             "wrong_address": "The package went to the wrong place", "damaged_in_transit": "The package arrived damaged",
             "other": "A shipping problem that fits none of the above"}},
         "requested_resolution": {"type": "choice", "instructions": "What does the customer want to happen?", "criteria": {
             "exchange": "Swap the item for a different one", "refund": "Money back",
             "replacement": "The same item sent again", "information": "Just an answer, no action needed"}},
         "tone": {"type": "choice", "instructions": "What is the customer's tone?",
                  "criteria": {"calm": None, "frustrated": None, "angry": None}},
     },
     {"department": ("choice", "returns"), "return_reason": ("choice", "wrong_size"),
      "shipping_issue": ("choice", "delayed"), "requested_resolution": ("choice", "refund"),
      "tone": ("choice", "frustrated")}),
    ("choice/structured-criteria",
     "I sent the shoes back a week ago. When do I get my money?",
     {"return_topic": {"type": "choice",
                       "instructions": {"question": "Which returns topic is the customer asking about?",
                                        "focus": "Classify the information the customer wants."},
                       "criteria": {
                           "return_policy": {"what": "Whether and how an item can be returned",
                                             "not_for": "Progress of a return already sent",
                                             "examples": ["Can I return shoes I've worn once?", "How long do I have to return an order?"]},
                           "return_status": {"what": "Progress of a return already sent",
                                             "not_for": "Whether and how an item can be returned",
                                             "examples": ["Has my return arrived yet?", "When will my refund be paid?"]}}}},
     {"return_topic": ("choice", "return_status")}),
    ("choice/taxonomy-subtrees",
     "32oz plastic bottle with a flip straw lid. Fits most bike cages.",
     {"department": {"type": "choice", "instructions": "Which top-level department does this product belong to?", "criteria": {
         "Sporting Goods": {"Cycling": ["Bike Bottles & Cages", "Bike Lights", "Helmets"],
                            "Fitness": ["Yoga Mats", "Resistance Bands"],
                            "Outdoor": ["Tents", "Sleeping Bags", "Hydration Packs"]},
         "Home & Kitchen": {"Drinkware": ["Water Bottles", "Travel Mugs", "Tumblers"],
                            "Cookware": ["Pots & Pans", "Bakeware"]},
         "Baby & Toddler": ["Sippy Cups", "Bottle Warmers", "Bibs"]}}},
     {"department": ("choice", "Sporting Goods")}),  # doc shows no number; plausible answer
]

SEVERITY_TABLE = [
    ("The export button is misaligned by a few pixels on the settings page.", 0.0),
    ("The PDF export button does nothing when clicked. I can still export to CSV and convert it myself, but that takes ages.", 1.0),
    ("Export to PDF fails with a spinner that never finishes. Some of our team say CSV export still works for them, others say it fails too.", 1.11),
    ("The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.", 1.43),
    ("Nobody on our team can log in since this morning. We get a 500 error on every attempt.", 2.0),
]
for i, (s, v) in enumerate(SEVERITY_TABLE):
    CASES.append((f"score/severity-{i}", s, {"sev": SEVERITY}, {"sev": ("score", v)}))

CASES.append(("score/numbers-only-antipattern",
              "The export button is misaligned by a few pixels on the settings page.",
              {"sev": {"type": "score", "instructions": "Rate severity from 0 to 2, where 2 is worst", "criteria": ["0", "1", "2"]}},
              {"sev": ("score", 0.55)}))

CASES.append(("score/composite-3q",
              "Export to PDF fails with a spinner that never finishes. Some of our team say CSV export still works for them, others say it fails too. This is the third time I'm writing in and honestly I'm done. Steps: open any report, click Export, choose PDF. Chrome 128 on macOS.",
              {"severity": SEVERITY,
               "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": [
                   "Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threatening to leave"]},
               "report_quality": {"type": "score", "instructions": "How much does the report give an engineer to work with?", "criteria": [
                   "No detail; just says something is broken", "Names the feature but no steps or environment",
                   "Steps to reproduce or environment, but not both", "Steps to reproduce and environment"]}},
              {"severity": ("score", 1.24), "frustration": ("score", 1.28), "report_quality": ("score", 3.0)}))

CASES.append(("score/structured-levels",
              "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
              {"sev": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": [
                  {"what": "Cosmetic; no impact to functionality", "examples": ["typo in a label", "misaligned icon"]},
                  {"what": "Broken or degraded feature, but workaround exists", "examples": ["export fails in one browser but works in another"]},
                  {"what": "Blocking issue; no workaround exists", "examples": ["cannot log in", "data loss"]}]}},
              {"sev": ("score", 1.03)}))

NOUL_TABLE = [
    ("Thanks, that fixed it!", 0.02), ("How do I reset my password?", 0.07),
    ("I need this sorted today, whatever it takes.", 0.26), ("Are you a bot?", 0.40),
    ("Is there any way to speak to someone about my invoice?", 0.84),
    ("I have asked three times now. Can I please just talk to a real person?", 0.99),
]
for i, (s, v) in enumerate(NOUL_TABLE):
    CASES.append((f"noul/escalation-{i}", s, {"esc": ESCALATION}, {"esc": ("noul", v)}))

CASES.append(("noul/criteria-true-false",
              "I have asked three times now. Can I please just talk to a real person?",
              {"rep": {"type": "noul", "instructions": "Has the customer contacted support about this before?",
                       "criteria": {"true": "Mentions a prior attempt, ticket, or that they have asked before",
                                    "false": "No sign of any previous contact"}}},
              {"rep": ("noul", 0.93)}))

RESUME = {"resume": {"name": "John Smith", "location": "Oakland, CA",
                     "summary": "Backend engineer with eight years of Python and Go experience.",
                     "experience": [{"employer": "Google", "title": "Senior Backend Engineer", "years": "2021-2025"},
                                    {"employer": "Microsoft", "title": "Software Engineer", "years": "2017-2021"}]}}
def dup(rec):
    return {"type": "noul", "instructions": {"potential_duplicate": rec,
            "question": "Is the resume for the same person as `potential_duplicate`?"}}
CASES.append(("noul/structured-instructions+json-state", RESUME,
              {"r18": dup({"name": "Jon Smith", "location": "Oakland, CA", "last_employer": "Google"}),
               "r42": dup({"name": "John Smith", "location": "Austin, TX", "last_employer": "Lone Star Freight"}),
               "r77": dup({"name": "John Smithers", "location": "Oakland, CA", "last_employer": "Bay Health Clinic"})},
              {"r18": ("noul", 0.74), "r42": ("noul", 0.09), "r77": ("noul", 0.08)}))

PY_Q = "Is the candidate strong in Python?"
PY_S = {"type": "score", "instructions": "How much Python experience does the candidate have?",
        "criteria": ["No experience", "Some familiarity", "Regular use in a job", "Deep expertise"]}
for i, (s, n, sc) in enumerate([
    ("My experience is in Java and Go. I have not used Python.", 0.03, 0.0),
    ("I have used Python occasionally for small scripts alongside my main Java work.", 0.14, 1.0),
    ("I used Python every day for two years in my last job, mostly data pipelines.", 0.81, 2.05),
    ("I have written Python daily for eight years, including maintaining a large Django codebase.", 0.92, 2.89),
]):
    CASES.append((f"noul-vs-score/python-{i}", s,
                  {"strong": {"type": "noul", "instructions": PY_Q}, "exp": PY_S},
                  {"strong": ("noul", n), "exp": ("score", sc)}))

CASES.append(("path-refs/backticks",
              {"ticket_message": "My flight was cancelled. Can I get a refund?",
               "refund_policy": "Cancelled flights are eligible for a full refund."},
              {"refund_requested": {"type": "noul", "instructions": "Does `ticket_message` request a refund?"},
               "request_type": {"type": "choice", "instructions": "What is the main request in `ticket_message`?", "criteria": {
                   "refund": "The customer wants money returned.", "rebooking": "The customer wants a replacement flight.",
                   "information": "The customer is asking for information only."}}},
              {"refund_requested": ("noul", 0.95), "request_type": ("choice", "refund")}))  # doc gives no numbers

CASES.append(("noul/structured-criteria",
              {"sender": {"display_name": "Beaver Dam Builders Ltd.", "email": "donotreply@payroll.example"},
               "message": "Your Q3 bonus is ready. Reply with your login password so we can verify your identity and release the funds."},
              {"cred": {"type": "noul", "instructions": {
                  "question": "Does the `message` ask the recipient to disclose a sensitive credential?", "inspect": "message",
                  "focus": "Look for a request to send the credential itself, not a request to change or reset it."},
                  "criteria": {"true": {"what": "Asks the recipient to reply with, type, or send a password, PIN, one-time code, or other security sensitive answer",
                                        "examples": ["Reply with your password", "Send us the 6-digit code you just received"]},
                               "false": {"what": "No sensitive credential is requested",
                                         "examples": ["Reset your password from the settings page", "Your statement is ready"]}}}},
              {"cred": ("noul", 0.95)}))  # doc gives no number


def agree(field, laya_v, jev_v):
    if field == "choice":
        return laya_v == jev_v
    tol = 0.25 if field == "noul" else 0.5
    return abs(laya_v - jev_v) <= tol


def run(model_id):
    agent = laya.load(model_id)
    ok = total = 0
    print(f"\n=== {model_id}")
    print(f"{'case':42} {'q':22} {'field':6} {'jev':>14} {'laya':>14}  conf  ok")
    for name, state, qs, ref in CASES:
        ans = agent.predict(state, qs)["answers"]
        for qid, (field, jev_v) in ref.items():
            a = ans[qid]
            v = a[field]
            hit = agree(field, v, jev_v)
            ok += hit
            total += 1
            print(f"{name:42} {qid:22} {field:6} {str(jev_v):>14} {str(v):>14}  {a['confidence']:.2f}  {'✓' if hit else '✗'}")
    print(f"agreement: {ok}/{total}  (choice exact, score ±0.5, noul ±0.25)")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["aac6fef/laya-mlx", "aac6fef/laya-multilingual-mlx", "aac6fef/laya-typed-decisions-mlx"]:
        run(m)
