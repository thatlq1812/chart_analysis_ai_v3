"""
Temporary audit script for Stage 3 extracted features quality check.
Run once: .venv/Scripts/python.exe scripts/_audit_stage3.py
"""
import json
import random
from pathlib import Path
from collections import defaultdict

feature_dir = Path("data/academic_dataset/stage3_features")
all_files = list(feature_dir.rglob("*.json"))
print(f"Total files scanned: {len(all_files)}")

bad = []
ok_files = []
for f in all_files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        ok_files.append((f, data))
    except Exception as e:
        bad.append((f, str(e)))

print(f"Valid JSON  : {len(ok_files)}")
print(f"Corrupted  : {len(bad)}")
print(f"Error rate : {len(bad)/len(all_files)*100:.1f}%")

if bad:
    print(f"\nSample corrupted files:")
    for f, err in bad[:5]:
        size = f.stat().st_size
        print(f"  {f.parent.name}/{f.name} ({size:,} bytes) -- {err[:60]}")

# ---- Quality check on valid files ----
print(f"\n=== QUALITY CHECK ON {len(ok_files)} VALID FILES ===")

sample = random.sample(ok_files, min(200, len(ok_files)))

axis_populated = 0       # x_range[0] is not None
axis_conf_total = 0.0
texts_total = 0
elements_total = 0
axis_x_none = 0
axis_field_present = 0

chart_type_dist = defaultdict(int)

for f, data in sample:
    ct = data.get("chart_type", "unknown")
    chart_type_dist[ct] += 1

    ai = data.get("axis_info")
    if ai is not None:
        axis_field_present += 1
        xr = ai.get("x_range")
        if xr and xr[0] is not None:
            axis_populated += 1
        else:
            axis_x_none += 1

    conf = data.get("confidence") or {}
    axis_conf_total += conf.get("axis_calibration_confidence", 0)

    texts_total += len(data.get("texts") or [])
    elements_total += len(data.get("elements") or [])

n = len(sample)
print(f"\nChart type distribution:")
for ct, cnt in sorted(chart_type_dist.items()):
    print(f"  {ct:12s}: {cnt}")

print(f"\nField quality:")
print(f"  axis_info field present  : {axis_field_present}/{n} ({axis_field_present/n*100:.0f}%)")
print(f"  axis x_range populated   : {axis_populated}/{n} ({axis_populated/n*100:.0f}%)")
print(f"  axis x_range = None      : {axis_x_none}/{n} ({axis_x_none/n*100:.0f}%)")
print(f"  avg axis_cal_confidence  : {axis_conf_total/n:.3f}")
print(f"  avg texts per chart      : {texts_total/n:.1f}")
print(f"  avg elements per chart   : {elements_total/n:.1f}")

# ---- Deep dive: what does axis_info look like? ----
print(f"\n=== AXIS_INFO SAMPLE (first 5 with axis_info) ===")
shown = 0
for f, data in ok_files:
    ai = data.get("axis_info")
    if ai:
        print(f"\n  File: {f.parent.name}/{f.name}")
        print(f"  chart_type : {data.get('chart_type')}")
        print(f"  x_range    : {ai.get('x_range')}")
        print(f"  y_range    : {ai.get('y_range')}")
        print(f"  x_labels   : {ai.get('x_labels', [])[:5]}")
        print(f"  y_labels   : {ai.get('y_labels', [])[:5]}")
        print(f"  x_cal_conf : {ai.get('x_calibration_confidence', 0):.3f}")
        print(f"  y_cal_conf : {ai.get('y_calibration_confidence', 0):.3f}")
        shown += 1
        if shown >= 5:
            break

if shown == 0:
    print("  (No files with axis_info found)")
