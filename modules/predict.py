import pandas as pd
from pathlib import Path
import dill
import json
from datetime import datetime


def load_model():
    models_dir = Path("data/models")
    model_files = list(models_dir.glob("*.pkl"))

    if not model_files:
        raise FileNotFoundError("Файлы моделей не найдены")

    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    model_path = model_files[0]

    with open(model_path, 'rb') as f:
        return dill.load(f)


def load_test_data():
    test_dir = Path("data/test")
    json_files = list(test_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("Тестовые файлы не найдены")

    test_data = []
    for json_file in json_files:
        try:
            df = pd.read_json(json_file)
        except:
            try:
                df = pd.read_json(json_file, lines=True)
            except:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    df = pd.DataFrame([data])
        test_data.append(df)

    return test_data


def predict():

    model = load_model()
    test_data = load_test_data()
    all_predictions = []

    for i, df in enumerate(test_data):
        df_processed = df.copy()

        if 'year' in df_processed.columns:
            current_year = datetime.now().year
            df_processed['car_age'] = current_year - df_processed['year']
        predictions = model.predict(df_processed)

        pred_df = pd.DataFrame({
            'file_index': i,
            'row_index': range(len(predictions)),
            'prediction': predictions
        })

        if 'price' in df.columns:
            pred_df['original_price'] = df['price'].values
        if 'year' in df.columns:
            pred_df['original_year'] = df['year'].values
        if 'manufacturer' in df.columns:
            pred_df['original_manufacturer'] = df['manufacturer'].values
        if 'model' in df.columns:
            pred_df['original_model'] = df['model'].values

    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
    else:
        combined_predictions = pd.DataFrame()
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"predictions_{timestamp}.csv"

    combined_predictions.to_csv(output_file, index=False)


    return str(output_file)


if __name__ == "__main__":
    predict()
