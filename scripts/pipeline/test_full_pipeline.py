#!/usr/bin/env python
"""
Full Pipeline Test - Stage 3 Extraction

Test Stage 3 với synthetic chart có ground truth.
Chạy: .venv/Scripts/python.exe scripts/pipeline/test_full_pipeline.py
"""

import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io
import json

# ============================================================
# GROUND TRUTH
# ============================================================
GROUND_TRUTH = {
    'chart_type': 'bar',
    'title': 'Monthly Sales 2025',
    'x_label': 'Month',
    'y_label': 'Revenue (USD)',
    'x_values': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'y_values': [1200, 1800, 1500, 2200, 1900],
    'bar_color': '#4472C4',
}


def create_bar_chart(data: dict) -> np.ndarray:
    """Create bar chart and return as BGR numpy array."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    x = np.arange(len(data['x_values']))
    bars = ax.bar(x, data['y_values'], color=data['bar_color'], width=0.6)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, data['y_values']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel(data['x_label'], fontsize=12)
    ax.set_ylabel(data['y_label'], fontsize=12)
    ax.set_title(data['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data['x_values'])
    ax.set_ylim(0, max(data['y_values']) * 1.2)
    ax.yaxis.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    
    if img_array.shape[-1] == 4:
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def main():
    print('=' * 60)
    print('FULL PIPELINE TEST - Stage 3 Extraction')
    print('=' * 60)
    
    # Step 1: Create synthetic chart
    print('\n[Step 1] Create Synthetic Chart')
    print('-' * 40)
    for k, v in GROUND_TRUTH.items():
        print(f'  {k}: {v}')
    
    test_image = create_bar_chart(GROUND_TRUTH)
    print(f'\nImage created: {test_image.shape}')
    
    # Save for reference
    save_path = Path('data/samples/test_bar_chart.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), test_image)
    print(f'Saved to: {save_path}')
    
    # Step 2: Run Stage 3
    print('\n[Step 2] Stage 3 Extraction')
    print('-' * 40)
    
    from core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig
    
    config = ExtractionConfig(
        ocr_engine='paddleocr',
        use_resnet_classifier=False,
        use_ml_classifier=False,
        enable_classification=False,
        enable_vectorization=False,
        enable_element_detection=True,
        enable_ocr=True,
    )
    config.ocr.use_gpu = False
    config.ocr.use_cache = False
    
    stage3 = Stage3Extraction(config)
    result = stage3.process_image(test_image, chart_id='test_001')
    
    print(f'Chart Type: {result.chart_type}')
    print(f'Texts found: {len(result.texts)}')
    print(f'Elements found: {len(result.elements)}')
    
    # Step 3: Evaluation
    print('\n[Step 3] Evaluation')
    print('-' * 40)
    
    detected_texts = [t.text for t in result.texts]
    
    # Title
    title_ok = any(GROUND_TRUTH['title'].lower() in t.lower() for t in detected_texts)
    print(f'[{"OK" if title_ok else "FAIL"}] Title detection')
    
    # X labels
    x_found = sum(1 for lbl in GROUND_TRUTH['x_values'] 
                  if any(lbl.lower() in t.lower() for t in detected_texts))
    x_ok = x_found == len(GROUND_TRUTH['x_values'])
    print(f'[{"OK" if x_ok else "PARTIAL"}] X labels: {x_found}/{len(GROUND_TRUTH["x_values"])}')
    
    # Y values
    detected_nums = set()
    for t in detected_texts:
        try:
            detected_nums.add(int(t.replace(',', '').strip()))
        except:
            pass
    y_found = sum(1 for v in GROUND_TRUTH['y_values'] if v in detected_nums)
    y_ok = y_found == len(GROUND_TRUTH['y_values'])
    print(f'[{"OK" if y_ok else "PARTIAL"}] Y values: {y_found}/{len(GROUND_TRUTH["y_values"])}')
    
    # Bar count
    bar_elements = [e for e in result.elements if e.element_type == 'bar']
    bar_ok = len(bar_elements) == len(GROUND_TRUTH['y_values'])
    print(f'[{"OK" if bar_ok else "FAIL"}] Bars: {len(bar_elements)}/{len(GROUND_TRUTH["y_values"])}')
    
    # Summary
    print('\n' + '=' * 60)
    passed = sum([title_ok, x_ok, y_ok, bar_ok])
    print(f'SCORE: {passed}/4 ({100*passed/4:.0f}%)')
    print('=' * 60)
    
    # Detailed output
    print('\n[Detected Texts]')
    for i, t in enumerate(result.texts[:15]):
        role = t.role.value if hasattr(t.role, 'value') else str(t.role) if t.role else '-'
        print(f'{i+1:2}. [{role:12}] {t.confidence:.0%} "{t.text}"')
    
    return passed >= 3  # Pass if 3/4 tests OK


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
