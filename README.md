# ❤️ Heart Disease Predictor

A Flask-based web application that predicts heart disease risk using an ensemble of machine learning models. Supports both **manual input** and **AI-powered report file upload** (via OpenAI GPT).

---

## 🚀 Features

- Predict heart disease risk from 13 clinical features
- Upload a patient report (`.txt` or `.csv`) — GPT extracts features automatically
- Ensemble of 3 ML models: Logistic Regression, Random Forest, KNN
- Displays per-model probability, average risk score, and model accuracy comparison
- Clean, responsive UI built with Flask + Jinja2 templates

---

## 🗂️ Project Structure

heart-disease-predictor/
│
├── app.py                  # Flask app — routes, GPT feature extraction, prediction
├── train_model.py          # Train & save model artifact (model.pkl)
├── evaluate_model.py       # Evaluate models — metrics, plots, confusion matrix
├── heart.csv               # Dataset (Cleveland Heart Disease dataset)
├── requirements.txt        # Python dependencies
│
├── templates/
│   ├── index.html          # Input form (manual + file upload)
│   └── result.html         # Prediction results page
│
└── static/
└── styles.css          # App styling

---

## 🧠 ML Models Used

| Model               | Algorithm              |
|---------------------|------------------------|
| Logistic Regression | Linear classifier       |
| Random Forest       | Ensemble (100 trees)   |
| KNN                 | K-Nearest Neighbors (k=5) |

All models are trained on the **Cleveland Heart Disease dataset** with `StandardScaler` normalization.

---

## 📊 Input Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = Yes) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible) |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/akshitasankhwar/Predictive-Heart-Health-An-AI-approach.git
cd heart-disease-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key

Create a `.env` file in the root directory:

OPENAI_API_KEY=your_openai_api_key_here

### 4. Train the model

```bash
python train_model.py
```

This generates `model.pkl` which the Flask app loads at runtime.

### 5. (Optional) Evaluate model performance

```bash
python evaluate_model.py
```

Outputs confusion matrices, ROC curves, and a performance CSV to the `results/` folder.

### 6. Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🖥️ Usage

### Option 1 — Manual Input
Fill in the 13 clinical fields in the form and click **"Predict from Manual Input"**.

### Option 2 — Upload a Report File
Upload a `.txt` or `.csv` file containing patient health information in plain text. GPT (`gpt-4o-mini`) will automatically extract the 13 required features and run the prediction.

---

## 📈 Evaluation Metrics

Running `evaluate_model.py` generates the following in the `results/` folder:

- `model_performance.csv` — Accuracy, Precision, Recall, F1, ROC AUC, Specificity
- `confusion_matrix_<model>.png` — Per-model confusion matrix
- `roc_curve.png` — ROC curve comparison across all models
- `model_accuracy_comparison.png` — Bar chart of model accuracies
- `feature_importance.png` — Random Forest feature importances
- `correlation_heatmap.png` — Feature correlation heatmap

---

## 🔧 Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (Logistic Regression, Random Forest, KNN)
- **AI Feature Extraction:** OpenAI GPT-4o-mini
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** joblib

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It is not intended to be used as medical advice or a diagnostic tool. Always consult a qualified healthcare professional.

---

## 📄 License

MIT License


