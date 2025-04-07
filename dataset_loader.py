import pandas as pd
from pathlib import Path
import json
import requests
from .config import DATASET_URL, DATASET_COLUMNS


def load_dataset():
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "paranormal_reports.csv"
    processed_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.json"

    # 1. Download dataset if missing
    if not raw_path.exists():
        raw_path.parent.mkdir(exist_ok=True)
        print("⏳ Downloading dataset...")
        try:
            response = requests.get(DATASET_URL)
            response.raise_for_status()
            with open(raw_path, 'wb') as f:
                f.write(response.content)
            print("✅ Dataset downloaded")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return []

    # 2. Load and validate DataFrame
    try:
        df = pd.read_csv(raw_path, encoding='latin1')  # Explicitly define df here
        print(f"Loaded dataset with {len(df)} entries")

        # Validate columns
        required_cols = {DATASET_COLUMNS['text_column'], DATASET_COLUMNS['title_column']}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return []

        # 3. Process data
        reports = []
        for _, row in df.iterrows():  # Now df is defined
            reports.append({
                "text": str(row[DATASET_COLUMNS['text_column']]),
                "title": str(row[DATASET_COLUMNS['title_column']]),
                "location": str(row.get(DATASET_COLUMNS['location_column'], 'Unknown')),
                "type": str(row.get(DATASET_COLUMNS['type_column'], 'Unspecified')),
                "label": "paranormal" if "ghost" in str(
                    row.get(DATASET_COLUMNS['type_column'], '')).lower() else "normal"
            })

        # 4. Save processed data
        processed_path.parent.mkdir(exist_ok=True)
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

        print(f"✅ Processed {len(reports)} reports")
        return reports

    except Exception as e:
        print(f"❌ Processing error: {e}")
        return []
