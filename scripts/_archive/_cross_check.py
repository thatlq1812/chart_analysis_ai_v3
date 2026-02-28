"""
Cross-check Stage 3 features vs QA pairs to verify data consistency.
Run: .venv/Scripts/python.exe scripts/_cross_check.py
"""
import json
import random
from pathlib import Path

FEATURE_DIR = Path("data/academic_dataset/stage3_features")
QA_DIR = Path("data/academic_dataset/chart_qa_v2/generated")
SLM_TRAIN = Path("data/slm_training/train.json")  # try v1 path first

CHART_TYPES = ["bar", "pie", "box", "histogram", "heatmap", "scatter"]

# ── 1. Cross-check Stage3 feature vs QA pair ────────────────────────────────
print("=" * 70)
print("SECTION 1: Stage3 features vs QA pairs (per chart type)")
print("=" * 70)

match_count = 0
no_qa_count = 0

for ct in CHART_TYPES:
    feat_files = list((FEATURE_DIR / ct).glob("*.json"))
    if not feat_files:
        print(f"\n[{ct.upper()}] No features yet")
        continue

    # Find file with matching QA
    random.shuffle(feat_files)
    sample = None
    for f in feat_files[:30]:
        qa_file = QA_DIR / ct / f"{f.stem}.json"
        if qa_file.exists():
            sample = (f, qa_file)
            match_count += 1
            break
        else:
            no_qa_count += 1

    if not sample:
        print(f"\n[{ct.upper()}] No QA file found for sampled features")
        continue

    feat_f, qa_f = sample
    feat = json.loads(feat_f.read_text(encoding="utf-8"))
    qa = json.loads(qa_f.read_text(encoding="utf-8"))

    ai = feat.get("axis_info") or {}
    texts = [t["text"] for t in (feat.get("texts") or [])]
    elems = feat.get("elements") or []
    conf = feat.get("confidence") or {}

    qa_pairs = qa.get("qa_pairs") or qa.get("questions") or []
    caption = qa.get("caption") or qa.get("description") or "(no caption)"

    print(f"\n[{ct.upper()}] {feat_f.stem}")
    print(f"  -- Stage3 features --")
    print(f"  OCR texts ({len(texts)}): {texts[:8]}")
    print(f"  Elements detected: {len(elems)}")
    print(f"  axis x_range: [{ai.get('x_min')}, {ai.get('x_max')}]")
    print(f"  axis y_range: [{ai.get('y_min')}, {ai.get('y_max')}]")
    print(f"  axis x_cal_conf: {ai.get('x_calibration_confidence', 0):.3f}")
    print(f"  axis y_cal_conf: {ai.get('y_calibration_confidence', 0):.3f}")
    print(f"  ocr_conf: {conf.get('ocr_mean_confidence', 0):.3f}")
    print(f"  -- QA pairs --")
    print(f"  caption: {str(caption)[:120]}")
    print(f"  total qa_pairs: {len(qa_pairs)}")
    for i, q in enumerate(qa_pairs[:4], 1):
        qtext = q.get("question", "")
        ans = q.get("answer", "")
        qtype = q.get("question_type") or q.get("type", "?")
        print(f"  [{i}] [{qtype}] Q: {qtext[:75]}")
        print(f"       A: {str(ans)[:60]}")

# ── 2. SLM training sample check ────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 2: SLM training dataset samples (slm_training_v2/train.json)")
print("=" * 70)

# Try multiple possible paths
for slm_path in [
    Path("data/slm_training/train.json"),
    Path("data/slm_training_v2/train.json"),
    Path("data/slm_training_v3/train.json"),
]:
    if slm_path.exists():
        SLM_TRAIN = slm_path
        break

if SLM_TRAIN.exists():
    raw = json.loads(SLM_TRAIN.read_text(encoding="utf-8"))
    # Handle both list-of-dicts and HF datasets format
    samples = raw if isinstance(raw, list) else raw.get("data", [])
    print(f"Total train samples: {len(samples)}")

    # Sample 5 and show instruction/output structure
    # Print all keys in first sample to understand schema
    first = samples[0]
    print(f"  Keys: {list(first.keys())}")
    shown = random.sample(samples, min(5, len(samples)))
    for i, s in enumerate(shown, 1):
        # Try common field names
        instruction = s.get("instruction") or s.get("prompt") or s.get("input") or ""
        output = s.get("output") or s.get("response") or s.get("answer") or ""
        chart_type = s.get("chart_type") or s.get("metadata", {}).get("chart_type", "?")
        has_axis = "axis_range" in str(instruction) or "x_range" in str(instruction)
        has_elements = "elements" in str(instruction)
        print(f"\n  [{i}] chart_type={chart_type}")
        print(f"  has_axis_in_prompt: {has_axis}  has_elements: {has_elements}")
        print(f"  instruction (first 200): {str(instruction)[:200]}")
        print(f"  output (first 150): {str(output)[:150]}")
else:
    print(f"  File not found: {SLM_TRAIN}")

# ── 3. Consistency check: do texts in Stage3 appear in QA answers? ──────────
print()
print("=" * 70)
print("SECTION 3: Value consistency - do OCR texts appear in QA answers?")
print("=" * 70)

consistent = 0
checked = 0

for ct in ["bar", "histogram", "scatter"]:
    feat_files = list((FEATURE_DIR / ct).glob("*.json"))
    random.shuffle(feat_files)
    for feat_f in feat_files[:15]:
        qa_f = QA_DIR / ct / f"{feat_f.stem}.json"
        if not qa_f.exists():
            continue
        feat = json.loads(feat_f.read_text(encoding="utf-8"))
        qa = json.loads(qa_f.read_text(encoding="utf-8"))
        texts = {t["text"] for t in (feat.get("texts") or [])}
        qa_text = " ".join(
            str(q.get("answer", "")) + " " + str(q.get("question", ""))
            for q in (qa.get("qa_pairs") or [])
        )
        # Check how many OCR tokens appear in QA text
        hits = sum(1 for t in texts if len(t) > 1 and t in qa_text)
        checked += 1
        if hits > 0:
            consistent += 1

print(f"  Checked: {checked} chart pairs (bar/histogram/scatter)")
print(f"  OCR text appears in QA: {consistent}/{checked} ({consistent/max(checked,1)*100:.0f}%)")
print(f"  (Expected: moderate overlap - QA uses semantic values, OCR is raw text)")
