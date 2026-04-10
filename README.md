# Spam Detector Frontend

This project adds a simple frontend to your notebook-based spam classifier.

## What it includes

- `app.py`: Streamlit web UI
- `model_backend.py`: Data cleaning, training, and prediction logic
- `smoke_test.py`: Quick CLI validation script
- `requirements.txt`: Python dependencies

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the frontend:

   ```bash
   streamlit run app.py
   ```

3. Optional quick backend check:

   ```bash
   python smoke_test.py
   ```

## Notes

- The app reads `combined_dataset.csv` from the project root.
- Labels are normalized and cleaned to avoid `NaN` target issues.
- Messages are cleaned using the same regex strategy as your notebook.
