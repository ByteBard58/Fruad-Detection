# 🛡️ FraudGuard Batch Analyzer

A machine learning–powered web application that detects fraudulent credit card transactions in bulk using a robust **XGBoost** model.
Built with **Flask**, **Scikit-Learn**, and **Pandas**, this project provides a sophisticated and automated solution for financial risk analysis.
This project demonstrates the power of self-healing systems and advanced machine learning in cybersecurity applications.
___

## 🚀 Overview

**FraudGuard Batch Analyzer** enables users to upload CSV files containing anonymized transaction data. It uses a trained **XGBoost Classifier** optimized via tailored research to predict if a transaction is **Fraudulent** or **Valid**.
The system includes:
- **Interactive Dashboard:** Visualizes risk statistics and high-risk transactions.
- **Batch Processing:** Handles large datasets instantly.
- **Self-Healing:** Automatically regenerates model artifacts if they are missing.

---

## 📸 Screenshots
*(Screenshots coming soon)*
<!-- 
![Home Page](static/images/ss_home.png)
![Dashboard](static/images/ss_dashboard.png)
-->

---

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection Dataset 2023 – Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **Classes:**
  - `0 → Valid Transaction`
  - `1 → Fraudulent Transaction`
> The dataset is artificially balanced. That is why, the number of 0 and 1 classes are equal.
- **Features:**
  - `V1-V28`: Anonymized features.
  - `Amount`: Transaction amount.

---

## ⚙️ Model Architecture

| Step | Description |
| :--- | :--- |
| **Imputation** | Missing values handled using median strategy |
| **Scaling** | Standardized with `StandardScaler` |
| **Dimensionality Reduction** | `PCA` (n_components=24) |
| **Classifier** | `XGBClassifier` (n_estimators=600, max_depth=10, learning_rate=0.1) |

Final model artifacts are serialized with `joblib` as:
```
models/
├── pipe.pkl
├── feat_names.pkl
```
---

## 🧪 Model Selection Research

We rigorously tested multiple algorithms including **Random Forest**, **SVC**, and **XGBoost** to find the optimal architecture. Using `RandomizedSearchCV`, we identified that XGBoost with PCA feature extraction yielded the best balance of speed and accuracy.

Here is the **Classification Report** for the final model:

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |        1.00 |     1.00 |       1.00 |      5000 |
| 1            |        1.00 |     1.00 |       1.00 |      5000 |
| accuracy     |        1.00 |     1.00 |       1.00 |      1.00 |
| macro avg    |        1.00 |     1.00 |       1.00 |     10000 |
| weighted avg |        1.00 |     1.00 |       1.00 |     10000 |

*Note: The model achieves near-perfect scores on the test set, demonstrating extreme robustness.*

You can find the detailed research code in the [research.py](research.py) file included in the repo. However, for the best viewing experience, use the HTML copy of the notebook which is available in the [research.html](reports/research.html) file.

---

## 🧩 Project Structure
```
FRAUD DETECTION/
├── Dataset/
│   └── creditcard_2023.csv  # Primary dataset
├── models/
│   ├── feat_names.pkl       # Serialized feature names
│   └── pipe.pkl             # Serialized machine learning pipeline
├── processed/               # Directory for analyzed output files (Ignored)
├── static/                  # Static assets for the web application
├── templates/
│   ├── index.html           # Upload page
│   └── dashboard.html       # Results dashboard
├── uploads/                 # Temporary storage for user uploads (Ignored)
|
├── .gitignore               # Files to exclude from version control
├── app.py                   # Main Flask application file
├── fit.py                   # Script for training and saving the model
├── LICENSE                  # Licensing information
├── research.py              # Marimo notebook for model research
└── requirements.txt         # Python package dependencies
```
---
## 💻 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ByteBard58/Fruad-Detection
cd "Fruad Detection"
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Configure Environment
Create a `.env` file in the root directory:
```bash
DATA_PATH="Dataset/creditcard_2023.csv"
```
### 4️⃣ Run the App
```bash
python app.py
```

### 5️⃣ Run Marimo Notebooks (Optional)
To explore the research process interactively:
```bash
marimo edit research.py
```
This command will open the notebook in your default browser.

---

## 🐳 Run the app directly via Dockerhub Image
*Coming Soon*

---

## 🌠 Web Interface

Users upload a CSV file containing transaction data. The system:
- Validates the columns.
- Processes the file using the pre-trained pipeline.
- Generates a **Risk Dashboard** with key insights.
- Allows downloading of the processed file with risk probability scores appended.

**Note:** The sophisticated UI/UX design was implemented with assistance from modern AI coding tools to ensure a premium user experience.

---

## 🧰 Tech Stack

- **Languages**: Python, HTML, CSS, JavaScript
- **Libraries**: Flask, Scikit-Learn, Pandas, NumPy, XGBoost, Joblib, Marimo
- **Dataset Source**: Kaggle Credit Card Fraud Detection

---

## 🪐 Author

Sakib ( ByteBard58 )

> Student | Aspiring Computer Engineer | AI & ML Enthusiast

📍 GitHub Profile: [ByteBard58](http://www.github.com/ByteBard58)

---

## 😃 Appreciation
I appreciate you taking the time to look over my work. I hope you found it interesting and enjoyable. If you could star it on GitHub, it would be really appreciated. 🌟

Do not hesitate to contact us if you have any queries, recommendations, or topics you would want to talk about. My [GitHub profile page] (http://www.github.com/ByteBard58) has my contact details.

Have a great day !
