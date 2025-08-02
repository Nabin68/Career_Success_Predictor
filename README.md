# 💼 Career Success Predictor

A machine learning-powered web app to estimate your salary based on factors like age, gender, education, experience, and job title. Built using **scikit-learn** and **Streamlit**.

---

## 🚀 Features

- Predicts salary based on user input
- Extracts **career level** from job title automatically
- One-hot encodes categorical features
- Scales numeric features for better model performance
- Fully interactive Streamlit UI

---

## 📁 Project Structure

```
.
├── datasets/
│   └── Salary_prediction.csv       # Input dataset
├── model/
│   └── train_model.py              # Training script
├── model.pkl                       # Trained regression model
├── scaler.pkl                      # StandardScaler for numeric features
├── encoder.pkl                     # OneHotEncoder for categorical features
├── main.py                         # Streamlit app (frontend)
├── requirements.txt                # Project dependencies
└── .gitignore                      # Files/folders to ignore (e.g., venv/)
```

---

## 🧠 Model Training

### ▶️ Run training script

```bash
python model/train_model.py
```

After training, the following files will be generated:

- `model.pkl` – Linear regression model
- `scaler.pkl` – Fitted `StandardScaler`
- `encoder.pkl` – Fitted `OneHotEncoder`

These are loaded by `main.py` to make predictions in the Streamlit app.

---

## 🌐 Running the App

### ▶️ Run Streamlit app

```bash
streamlit run main.py
```

Open the app in your browser and fill in your details to get an estimated salary.

---

## 📝 Notes

- The `"Job Title"` is processed into a new feature called **Career_Level**, categorized into:
  - `Junior`
  - `Mid`
  - `Senior`
- Categorical features are encoded with `OneHotEncoder(drop='first')` to avoid the dummy variable trap.
- The `"Salary"` column is moved to the end of the dataset for cleaner formatting before training.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🛑 .gitignore

Make sure to ignore the `venv/` folder and any unnecessary files:

```
venv/
*.pyc
__pycache__/
```

---

## 📬 Contact

For questions, suggestions, or improvements, feel free to open an issue or reach out via GitHub.

---

---
© 2025 | Developed by **Nabin** | All rights reserved.

