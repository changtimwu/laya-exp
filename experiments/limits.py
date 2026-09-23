"""Probe Laya's structural limits against Jev's documented ones (255 options, 10 levels).
Run: uv run experiments/limits.py
"""
import json
import warnings

import laya_mlx as laya

warnings.filterwarnings("ignore", category=RuntimeWarning)
agent = laya.load("aac6fef/laya-mlx")
print("max_len", agent.cfg.get("max_len"), "head_max_len", agent.cfg.get("head_max_len"))

# 1. How many Choice options fit? Correct answer is always one specific country.
COUNTRIES = ("Afghanistan Albania Algeria Andorra Angola Argentina Armenia Australia Austria Azerbaijan Bahamas Bahrain "
             "Bangladesh Barbados Belarus Belgium Belize Benin Bhutan Bolivia Botswana Brazil Brunei Bulgaria Burundi "
             "Cambodia Cameroon Canada Chad Chile China Colombia Comoros Congo Croatia Cuba Cyprus Czechia Denmark "
             "Djibouti Dominica Ecuador Egypt Eritrea Estonia Eswatini Ethiopia Fiji Finland France Gabon Gambia Georgia "
             "Germany Ghana Greece Grenada Guatemala Guinea Guyana Haiti Honduras Hungary Iceland India Indonesia Iran "
             "Iraq Ireland Israel Italy Jamaica Japan Jordan Kazakhstan Kenya Kiribati Kuwait Kyrgyzstan Laos Latvia "
             "Lebanon Lesotho Liberia Libya Liechtenstein Lithuania Luxembourg Madagascar Malawi Malaysia Maldives Mali "
             "Malta Mauritania Mauritius Mexico Moldova Monaco Mongolia Montenegro Morocco Mozambique Myanmar Namibia "
             "Nauru Nepal Netherlands Nicaragua Niger Nigeria Norway Oman Pakistan Palau Panama Paraguay Peru "
             "Philippines Poland Portugal Qatar Romania Russia Rwanda Samoa Senegal Serbia Seychelles Singapore "
             "Slovakia Slovenia Somalia Spain Sudan Suriname Sweden Switzerland Syria Tajikistan Tanzania Thailand Togo "
             "Tonga Tunisia Turkey Turkmenistan Tuvalu Uganda Ukraine Uruguay Uzbekistan Vanuatu Venezuela Vietnam "
             "Yemen Zambia Zimbabwe").split()
state = "I just got back from Kyoto and Osaka; the sushi was unbelievable and the bullet train was so fast."
print("\n# Choice option capacity (target = Japan)")
for n in [5, 10, 20, 40, 60, 80, 100, 150, len(COUNTRIES)]:
    opts = [c for c in COUNTRIES if c != "Japan"][: n - 1] + ["Japan"]
    try:
        a = agent.predict(state, {"q": {"type": "choice", "instructions": "Which country did the person visit?",
                                        "criteria": opts}})["answers"]["q"]
        print(f"n={n:3}  choice={a['choice']:12} p={a['probabilities'][a['choice']]:.3f}  conf={a['confidence']:.2f}")
    except ValueError as e:
        print(f"n={n:3}  ERROR: {e}")

# 2. Score levels beyond 10 (Jev caps at 10).
print("\n# Score with 11 levels (0..10 pain scale)")
levels = ["no pain at all", "barely noticeable", "minor, easy to ignore", "noticeable but not distracting",
          "distracting but manageable", "hard to ignore for long", "interferes with concentration",
          "limits daily activities", "hard to function", "unable to function, near unbearable",
          "worst pain imaginable, emergency"]
a = agent.predict("My back hurts a bit when I bend down, but I forget about it most of the day.",
                  {"q": {"type": "score", "instructions": "How severe is the pain described?", "criteria": levels}})
print(json.dumps({k: a["answers"]["q"][k] for k in ("score", "confidence")}))

# 3. State truncation: the answer sits at the end of a long state.
print("\n# Long state (evidence at the end)")
filler = "The weather today is mild and the meeting notes are unremarkable. " * 80
for label, s in [("evidence first", "Customer: please refund my duplicate charge. " + filler),
                 ("evidence last", filler + "Customer: please refund my duplicate charge.")]:
    a = agent.predict(s, {"q": {"type": "noul", "instructions": "Does the customer request a refund?"}})
    print(f"{label:15} noul={a['answers']['q']['noul']}  input_tokens={a['usage']['input_tokens']}")
