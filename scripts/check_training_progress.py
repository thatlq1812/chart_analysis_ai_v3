#!/usr/bin/env python3
"""Check YOLO training progress."""

from pathlib import Path
import json

RESULTS_DIR = Path("D:/elix/chart_analysis_ai_v3/results/training_runs/chart_detector_v3")

def main():
    print("=" * 60)
    print("YOLO TRAINING PROGRESS CHECK")
    print("=" * 60)
    
    # Check if training dir exists
    if not RESULTS_DIR.exists():
        print("Training not started yet or directory not found.")
        return
    
    # Check weights
    weights_dir = RESULTS_DIR / "weights"
    if weights_dir.exists():
        weights = list(weights_dir.glob("*.pt"))
        print(f"\nWeights saved: {len(weights)}")
        for w in weights:
            size_mb = w.stat().st_size / (1024 * 1024)
            print(f"  - {w.name}: {size_mb:.1f} MB")
    
    # Check results.csv
    results_csv = RESULTS_DIR / "results.csv"
    if results_csv.exists():
        lines = results_csv.read_text().strip().split("\n")
        print(f"\nTraining epochs completed: {len(lines) - 1}")
        
        if len(lines) > 1:
            header = lines[0].split(",")
            last_row = lines[-1].split(",")
            
            print("\nLatest metrics:")
            for h, v in zip(header[:8], last_row[:8]):
                print(f"  {h.strip()}: {v.strip()}")
    else:
        print("\nNo results.csv yet - training may still be scanning data.")
    
    # Check for plots
    plots = list(RESULTS_DIR.glob("*.png")) + list(RESULTS_DIR.glob("*.jpg"))
    if plots:
        print(f"\nPlots generated: {len(plots)}")
        for p in plots[:5]:
            print(f"  - {p.name}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
