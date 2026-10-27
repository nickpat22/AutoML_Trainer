# AutoML Trainer update

## Files in this pack
- `app.py` — refreshed Streamlit UI (Train, Compare, Predict, History tabs)
- `trainer.py` — classification/regression training and evaluation
- `predict.py` — discovers model input fields and predicts
- `database.py` — MongoDB history helpers
- `requirements.txt` — Python dependencies

## Install / update
1. Stop Streamlit with `Ctrl+C`.
2. Make a backup copy of your current `app.py`, `trainer.py`, `predict.py`, and `database.py`.
3. Copy the files from this folder into your existing `AutoML_Trainer` project folder, replacing those files.
4. Keep your existing `venv` and saved files in `models/`; they are not included in this pack.
5. Activate the virtual environment and run:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

MongoDB is configured for `mongodb://localhost:27017/` by default. You can override it with environment variables `MONGODB_URI` and `MONGODB_DATABASE`.

## Project structure

```text
AutoML_Trainer/
├── venv/                 # Keep your existing virtual environment
├── app.py                # Streamlit interface
├── trainer.py            # ML preprocessing, training, metrics
├── predict.py            # Generic prediction helpers
├── database.py           # MongoDB training history
├── requirements.txt
├── datasets/              # Optional datasets; uploaded data isn't copied here automatically
└── models/                # Saved .joblib pipelines; existing files stay here
```

The code was syntax-checked and the training/prediction functions were exercised against small numeric + categorical sample datasets for all listed algorithms. The full Streamlit UI was not run in this build environment, so install the requirements and test it with your local MongoDB before relying on it.

Only load `.joblib` model files that you created or otherwise trust. Joblib files can execute code when loaded.
