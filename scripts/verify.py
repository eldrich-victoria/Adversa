import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.pipeline import run_pipeline

inputs = [
    ("Clean toxic",         "you are an idiot"),
    ("Leetspeak",           "y0u r3 an 1d10t"),
    ("Spacing frag",        "i d i o t"),
    ("Repetition",          "stuuuuupid"),
    ("Unicode homoglyph",   "\u0443\u043e\u0443 \u0430\u0440\u0435 \u0430\u043d \u0456\u0434\u0456\u043e\u0442"),
    ("Benign",              "Have a great day!"),
]

for name, text in inputs:
    r = run_pipeline(text, k=5)
    clf = r["classification"]
    ai = r["attack_info"]
    nm = r["normalization"]
    print(f"[{name}]")
    print(f"  Input:    {text!r}")
    print(f"  Attacks:  {ai['attack_types']}")
    print(f"  Norm:     {nm['normalized']!r}")
    print(f"  Verdict:  {clf['label'].upper()} ({clf['confidence']:.0%})")
    print(f"  Category: {clf['category']}")
    print()
