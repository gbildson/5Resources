import re,sys
txt=sys.stdin.read()
pairs=re.findall(r"\"strict_score\":\s*([0-9.eE+-]+)\s*,\s*\"source_checkpoint\":\s*\"([^\"]+)\"", txt, flags=re.S)
rows=[(float(s), ckpt) for s,ckpt in pairs]
rows.sort(key=lambda x:x[0], reverse=True)
for score,ckpt in rows[:10]:
    print(f"{score:.6f}  {ckpt}")
