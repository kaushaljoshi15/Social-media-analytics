"""
Restore Full Sentiment140 Dataset (1.6 Million Rows / 238 MB)
════════════════════════════════════════════════════════════════
GitHub has a 100 MB file limit, so this dataset was compressed
and split into two parts (<41 MB each):
  - data/raw/sentiment140_full_part1.bin
  - data/raw/sentiment140_full_part2.bin

Run this script on any computer after cloning to instantly restore:
  data/raw/training.1600000.processed.noemoticon.csv (238 MB)
"""
import os
import zipfile
from pathlib import Path

def restore_sentiment140():
    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "raw"
    target_csv = raw_dir / "training.1600000.processed.noemoticon.csv"
    
    if target_csv.exists() and target_csv.stat().st_size > 200_000_000:
        print(f"✅ Dataset already exists ({target_csv.stat().st_size / 1024 / 1024:.1f} MB): {target_csv}")
        return

    part1 = raw_dir / "sentiment140_full_part1.bin"
    part2 = raw_dir / "sentiment140_full_part2.bin"
    temp_zip = raw_dir / "sentiment140_temp.zip"

    if not part1.exists() or not part2.exists():
        print("❌ Error: sentiment140_full_part1.bin or part2.bin not found in data/raw/")
        return

    print("🔄 Merging parts into zip...")
    with open(temp_zip, "wb") as fout:
        with open(part1, "rb") as f1:
            fout.write(f1.read())
        with open(part2, "rb") as f2:
            fout.write(f2.read())

    print("📦 Extracting training.1600000.processed.noemoticon.csv (238 MB)...")
    with zipfile.ZipFile(temp_zip, "r") as zf:
        zf.extractall(raw_dir)

    if temp_zip.exists():
        temp_zip.unlink()

    if target_csv.exists():
        size_mb = target_csv.stat().st_size / (1024 * 1024)
        print(f"🎉 Successfully restored full dataset ({size_mb:.1f} MB): {target_csv}")
    else:
        print("❌ Extraction failed.")

if __name__ == "__main__":
    restore_sentiment140()
