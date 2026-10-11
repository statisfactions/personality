"""Extract Hampson, Goldberg & John (1987) British SDY for the PDA-525 overlap.
OCR first:  pdftoppm -r 400 -f 6 -l 18 -gray Category-breadth.pdf pg; tesseract pg-NN.pgm pg-NN --psm 6
then set D to the OCR directory. Values cross-checked against the SDY-Z column; hand fixes from page images below.
"""
import re, glob, json, sys, difflib
sys.path.insert(0, "scripts"); import pkit
D = "/private/tmp/claude-501/-Users-rgb-src-personality/8424bf0d-5a3e-4f79-8664-314e1acb738a/scratchpad/hgj/"
zl = lambda v: 51.714 * v - 248.851
lines = {}
for f in sorted(glob.glob(D + "pg-*.txt")):
    for line in open(f):
        m = re.match(r"^[^a-zA-Z]*([a-z][a-z\-]*(?: [a-z\-]+)?)\s+(.*)$", line.strip())
        if m: lines.setdefault(m.group(1), (f[-6:-4], m.group(2)))
L = [a.lower() for a in pkit.load.adjectives()]
ok, manual, unmatched = {}, [], []
for a in L:
    if a not in lines:
        c = difflib.get_close_matches(a, list(lines), 1, 0.85)
        if c: unmatched.append((a, c[0]))
        continue
    pg, rest = lines[a]; rest2 = rest.replace("—", "-").replace("–", "-")
    toks = re.findall(r"-?\d+(?:[.,]\d+)?", rest2)
    dec = [float(t.replace(",", ".")) for t in toks if re.fullmatch(r"\d[.,]\d\d\d?", t)]
    v = None
    if len(toks) > 2 and re.fullmatch(r"\d[.,]\d\d\d?", toks[2]): v = float(toks[2].replace(",", ".")[:4])
    elif len(toks) > 2 and re.fullmatch(r"[1-9]\d\d", toks[2]): v = int(toks[2]) / 100
    elif dec: v = dec[0]
    zt = [int(t) for t in toks[3:4] if re.fullmatch(r"-?\d{1,4}", t)]
    agree = v is not None and zt and (abs(abs(zt[0]) - abs(zl(v))) <= 3)
    if agree: ok[a] = v
    else: ok[a] = v; manual.append((a, pg, rest, v, round(zl(v)) if v else None))
print(len(ok)-len(manual), "auto-verified;", len(manual), "need eyes;", "near-miss names:", unmatched)
for m in manual: print("  ", m)
json.dump(ok, open(D + "sdy_ok.json", "w"))
# hand-checked against page images (pp. 249, 251) or z-consistent dropped-digit OCR
ok.update({"funny": 7.11, "glad": 6.58, "interested": 7.08, "grumpy": 2.19, "sentimental": 5.25, "moral": 7.42})
alias = {"depressed": "a depressed", "nosy": "nosey", "prompt": "prompt -", "careful": "careful -", "easygoing": "easy-going",
         "ignorant": "i ignorant", "self-centered": "self-centred", "curious": "curious -", "expressive": "expressive -", "extraverted": "extroverted"}
for a, k in alias.items():
    pg, rest = lines[k]; print("alias", a, k, rest)
ok.update({"depressed": 2.00, "nosy": 3.00, "prompt": 6.91, "careful": 6.81, "easygoing": 7.06, "ignorant": 2.26,
           "self-centered": 2.83, "curious": 6.02, "expressive": 7.02, "extraverted": 6.06})
ok = {a: v for a, v in ok.items() if v is not None}
assert all(1 <= v <= 9 for v in ok.values())
import csv, os
os.makedirs("data/hampson_1987", exist_ok=True)
labs = pkit.load.adjectives()
with open("data/hampson_1987/sdy_british_pda525.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["adjective", "sdy_british"])
    for a in labs:
        if a.lower() in ok: wr.writerow([a, ok[a.lower()]])
print("wrote", len(ok), "overlap terms")
