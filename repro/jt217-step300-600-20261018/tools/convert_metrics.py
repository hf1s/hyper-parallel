import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = []
for line in src.read_text().splitlines():
    record = json.loads(line)
    rows.append({
        "step": int(record["step"]),
        "loss": float(record["loss"]),
        "grad_norm": float(record.get("grad_norm", "nan")),
        "loss_dict": {
            "foundation_loss/lm": float(record["components"]["foundation_loss/lm"]),
            "foundation_loss/mtp": float(record["components"]["foundation_loss/mtp"]),
        },
    })
dst.write_text("".join(json.dumps(row) + "\n" for row in rows))
print(f"{src} -> {dst}; rows={len(rows)}")
