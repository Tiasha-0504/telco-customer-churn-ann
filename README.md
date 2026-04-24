# 📡 Telco Customer Churn Prediction using ANN

A deep learning project that predicts whether a telecom customer will churn (leave the service) using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**. The project covers the full ML pipeline — from data preprocessing to model evaluation and single-sample prediction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [How to Run](#how-to-run)
- [Sample Prediction](#sample-prediction)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🔍 Overview

Customer churn is a critical problem for telecom companies. This project builds a binary classification model to predict whether a customer is likely to churn (`Yes`) or stay (`No`) based on their account details and usage patterns.

---

## 📊 Dataset

- **Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7,043 customers
- **Columns:** 21 features

### Key Features

| Feature            | Type        | Description                              |
|--------------------|-------------|------------------------------------------|
| `gender`           | Categorical | Customer gender                          |
| `SeniorCitizen`    | Binary      | Whether customer is a senior citizen     |
| `tenure`           | Numeric     | Number of months with the company        |
| `InternetService`  | Categorical | Type of internet service                 |
| `Contract`         | Categorical | Contract type (Month-to-month, etc.)     |
| `MonthlyCharges`   | Numeric     | Monthly billing amount                   |
| `TotalCharges`     | Numeric     | Total amount charged                     |
| `Churn`            | Target      | Whether the customer churned (Yes/No)    |

### Class Distribution

| Class      | Count |
|------------|-------|
| No Churn   | 5,174 |
| Churn      | 1,869 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Deep Learning:** TensorFlow 2.19.0 / Keras
- **Data Handling:** Pandas, NumPy
- **Preprocessing:** Scikit-learn (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`)
- **Visualization:** Matplotlib
- **Environment:** Google Colab / Jupyter Notebook

---

## 🔄 Project Workflow

1. **Data Loading** — Load CSV dataset from Google Drive
2. **Exploratory Data Analysis (EDA)** — Shape, dtypes, null values, churn distribution
3. **Data Cleaning**
   - Dropped `customerID` (irrelevant identifier)
   - Converted `TotalCharges` from `object` to `float64`
   - Filled 11 missing `TotalCharges` values with the **median**
4. **Feature Engineering**
   - Separated features `X` and target `y`
   - Encoded target with `LabelEncoder` → `No = 0`, `Yes = 1`
5. **Encoding Categorical Features**
   - Binary columns (e.g., `gender`, `Partner`) → `LabelEncoder`
   - Multi-class columns (e.g., `InternetService`, `Contract`) → `OneHotEncoder` with `drop='first'`
6. **Train-Test Split** — 80% train / 20% test (`random_state=0`)
7. **Feature Scaling** — `StandardScaler` applied to both train and test sets
8. **Model Building** — ANN with TensorFlow/Keras
9. **Model Training** — 100 epochs, batch size 32, 10% validation split
10. **Evaluation** — Confusion Matrix, Accuracy Score, Classification Report
11. **Single Prediction** — Predicts churn probability for a new customer

---

## 🧠 Model Architecture

```
Input: 29 features (after encoding)
│
├── Dense(16, activation='relu')
├── Dense(16, activation='relu')
├── Dense(8,  activation='relu')
└── Dense(1,  activation='sigmoid')   ← Binary Output

Optimizer : Adam
Loss      : Binary Crossentropy
Metric    : Accuracy
Epochs    : 100
Batch Size: 32
```

### Model Summary

| Layer       | Output Shape | Parameters |
|-------------|-------------|------------|
| Dense (ReLU)  | (None, 16)  | 480        |
| Dense (ReLU)  | (None, 16)  | 272        |
| Dense (ReLU)  | (None, 8)   | 136        |
| Dense (Sigmoid) | (None, 1) | 9          |
| **Total**   |             | **897**    |

---

## 📈 Results

| Metric               | Value  |
|----------------------|--------|
| **Test Accuracy**    | 76.08% |
| Precision (Churn)    | 0.55   |
| Recall (Churn)       | 0.51   |
| F1-Score (Churn)     | 0.53   |
| Precision (No Churn) | 0.83   |
| Recall (No Churn)    | 0.85   |
| F1-Score (No Churn)  | 0.84   |

### Confusion Matrix

```
              Predicted
              No     Yes
Actual No  [ 885    156 ]
       Yes [ 181    187 ]
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/telco-churn-ann.git
   cd telco-churn-ann
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib
   ```

3. **Add the dataset**
   - Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
   - Place it in the project folder or update the file path in the notebook

4. **Run the notebook**
   ```bash
   jupyter notebook Telco_Customer_Churn.ipynb
   ```
   Or open it directly in **Google Colab**.

---

## 🔮 Sample Prediction

```python
sample = pd.DataFrame({
    'gender': [1],
    'SeniorCitizen': [1],
    'Partner': [0],
    'Dependents': [0],
    'tenure': [2],
    'PhoneService': [1],
    'MultipleLines': [0],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': [1],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [75.5],
    'TotalCharges': [150.0]
})

sample_encoded = np.array(ct.transform(sample))
sample_scaled = sc.transform(sample_encoded)
prediction_prob = ann.predict(sample_scaled)
prediction = (prediction_prob > 0.5)[0][0]

print(f"Churn Probability: {prediction_prob[0][0]:.4f}")
# Output:
# Churn Probability: 0.7798
# ⚠ Prediction: Customer is likely to CHURN!
```

---

## 📁 Project Structure

```
telco-churn-ann/
│
├── Telco_Customer_Churn.ipynb              # Main Jupyter/Colab notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (download from Kaggle)
└── README.md                               # Project documentation
```

---

## 💡 Future Improvements

- Address **class imbalance** using SMOTE or class weights
- Experiment with **Dropout layers** to reduce overfitting
- Try **ensemble methods** like XGBoost or Random Forest for comparison
- Perform **hyperparameter tuning** with Keras Tuner
- Deploy the model as a **REST API** using Flask or FastAPI

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Made with ❤️ using TensorFlow & Python
