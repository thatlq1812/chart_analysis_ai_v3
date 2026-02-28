"""Full quality audit for all Stage3 extracted features."""
import json
from pathlib import Path
from collections import defaultdict

feature_dir = Path("data/academic_dataset/stage3_features")
all_files = list(feature_dir.rglob("*.json"))
print(f"Total files: {len(all_files)}")

bad = []
stats = defaultdict(list)

for f in all_files:
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        ct = d.get("chart_type", f.parent.name)
        ai = d.get("axis_info") or {}
        xr = ai.get("x_range")
        yr = ai.get("y_range")
        texts = d.get("texts") or []
        elems = d.get("elements") or []
        conf = (d.get("confidence") or {}).get("ocr_mean_confidence", 0)
        x_cal = ai.get("x_calibration_confidence", 0)
        y_cal = ai.get("y_calibration_confidence", 0)
        x_labels = ai.get("x_tick_labels") or ai.get("x_labels") or []
        y_labels = ai.get("y_tick_labels") or ai.get("y_labels") or []
        stats[ct].append({
            "has_axis": bool(ai),
            "x_range_ok": xr is not None and len(xr) == 2 and xr[0] is not None,
            "y_range_ok": yr is not None and len(yr) == 2 and yr[0] is not None,
            "has_x_labels": len(x_labels) > 0,
            "has_y_labels": len(y_labels) > 0,
            "text_count": len(texts),
            "elem_count": len(elems),
            "ocr_conf": conf,
            "x_cal": x_cal,
            "y_cal": y_cal,
        })
    except Exception as e:
        bad.append((f.name, str(e)))

print(f"Corrupted: {len(bad)}  Error rate: {len(bad)/max(len(all_files),1)*100:.2f}%")
if bad:
    for name, err in bad[:5]:
        print(f"  BAD: {name} -> {err}")

print()
header = f"{'Type':<12} {'N':>6}  {'axis%':>6}  {'x_rng%':>7}  {'y_rng%':>7}  {'xlbl%':>6}  {'ylbl%':>6}  {'txt':>5}  {'elm':>5}  {'ocr':>5}  {'x_cal':>5}  {'y_cal':>5}"
print(header)
print("-" * len(header))

total = 0
for ct in ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie", "area"]:
    rows = stats.get(ct, [])
    if not rows: continue
    n = len(rows)
    total += n
    def pct(key): return sum(1 for r in rows if r[key]) / n * 100
    def avg(key): return sum(r[key] for r in rows) / n
    print(
        f"{ct:<12} {n:>6}  {pct('has_axis'):>5.1f}%  {pct('x_range_ok'):>6.1f}%  {pct('y_range_ok'):>6.1f}%"
        f"  {pct('has_x_labels'):>5.1f}%  {pct('has_y_labels'):>5.1f}%"
        f"  {avg('text_count'):>5.1f}  {avg('elem_count'):>5.1f}"
        f"  {avg('ocr_conf'):>5.3f}  {avg('x_cal'):>5.3f}  {avg('y_cal'):>5.3f}"
    )

print(f"\nTotal valid: {total}/{len(all_files)}")

# -- Extra: check for charts with 0 texts or 0 elements (likely bad)
print()
for ct in ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie", "area"]:
    rows = stats.get(ct, [])
    if not rows: continue
    zero_text = sum(1 for r in rows if r["text_count"] == 0)
    zero_elem = sum(1 for r in rows if r["elem_count"] == 0)
    if zero_text or zero_elem:
        print(f"  [{ct}] zero_texts={zero_text}  zero_elements={zero_elem}")
