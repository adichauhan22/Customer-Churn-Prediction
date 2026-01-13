# 📊 Customer Churn Prediction — End-to-End Machine Learning Project

## 📌 Project Title & Overview

Customer churn is a critical challenge for subscription-based and service-driven businesses.  
This project builds an **end-to-end machine learning pipeline** to predict whether a customer is likely to churn based on historical behavior, service usage, and billing information.

The goal is to help businesses **identify at-risk customers early** and take proactive retention actions.

---

## 🎯 Problem Statement

Customer churn occurs when customers stop using a company’s product or service.  
High churn rates negatively impact revenue, customer lifetime value, and long-term growth.

**Objective:**  
To develop a machine learning model that predicts customer churn (`Yes / No`) using structured customer data.

---

## 💡 Why Churn Prediction Matters

- Retaining existing customers is significantly cheaper than acquiring new ones
- Early churn detection enables targeted retention strategies
- Helps businesses optimize pricing, contracts, and customer experience
- Supports data-driven decision-making for customer success teams

---

## 🧠 What This Project Demonstrates

This project demonstrates the complete lifecycle of a real-world ML project:

- Business understanding and problem framing
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Handling categorical and numerical features correctly
- Preventing data leakage using ML pipelines
- Training and evaluating multiple ML models
- Interpreting results from a business perspective

---

## 🗂 Dataset Information

- **Dataset:** Telco Customer Churn Dataset  
- **Size:** ~7,000 customer records  
- **Target Variable:** `Churn` (Yes / No)

### Feature Types
- **Numerical:** Tenure, MonthlyCharges, TotalCharges
- **Categorical:** Contract type, Payment method, Internet service, Add-on services

---

## 🔍 EDA Insights

Key insights discovered during exploratory data analysis:

- Customers with **short tenure** have a higher likelihood of churn
- **Month-to-month contracts** show significantly higher churn rates
- **Higher monthly charges** correlate strongly with churn
- Customers using **electronic check** as a payment method churn more frequently
- Long-term contracts and auto-payments improve customer retention

---

## ⚙️ Feature Engineering Summary

- Removed non-informative identifier columns (`customerID`)
- Converted target variable (`Churn`) into numerical format
- Handled missing values in numerical features
- Applied **Standard Scaling** to numerical features
- Applied **One-Hot Encoding** to categorical features
- Used `ColumnTransformer` to apply transformations correctly
- Performed stratified train–test split to avoid bias
- Prevented data leakage by fitting transformations only on training data
- Saved preprocessing pipeline for reuse and deployment

---

## 🤖 Models Trained

- **Logistic Regression** — baseline model for comparison
- **Random Forest Classifier** — captures non-linear relationships
- **Gradient Boosting Classifier** — final selected model due to superior performance

---

## 🏆 Final Model Selection

- **Final Model:** Gradient Boosting Classifier
- Selected based on highest **ROC-AUC score**
- ROC-AUC prioritized over accuracy due to churn class imbalance
- Balanced precision–recall trade-off suitable for churn prediction
- Effectively captures non-linear patterns in customer behavior

---

## 📈 Business Insights

- **Tenure** is one of the strongest predictors of churn, with new customers at highest risk
- **Contract type** heavily influences churn, especially month-to-month plans
- **Pricing sensitivity** plays a major role in customer churn
- **Payment method** impacts retention, with electronic check users more likely to churn
- Customers with **low tenure, high charges, and flexible contracts** should be targeted first
- Early retention strategies can significantly reduce revenue loss

---

## ⚠️ Limitations

- The model is trained on historical data and may require periodic retraining
- External factors such as competition or market changes are not included
- Class imbalance may affect optimal prediction thresholds
- Assumes all required features are available at prediction time

---

## 🏗️ Project Structure


Customer-Churn-Prediction/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   └── preprocessor.pkl
│
├── README.md
└── requirements.txt


---

## 🚀 How to Run the Project

## 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-link>
cd Customer-Churn-Prediction```


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebooks
Open Jupyter Notebook and execute notebooks in order:
01_eda.ipynb
02_feature_engineering.ipynb
03_model_training.ipynb
📈 Current Status
✅ EDA completed
✅ Feature engineering completed
🔄 Model training and evaluation in progress
🔜 Final model selection and performance metrics
🔜 Resume-ready results and deployment (optional)

🧪 Future Improvements :
Hyperparameter tuning for better performance
Model comparison using advanced algorithms
API deployment for real-time predictions
Monitoring and retraining strategies

👨‍💻 Author :
Aditya Chauhan
B.Tech Computer Science
Aspiring Machine Learning / AI Engineer

⭐ Why This Project Is Interview-Ready :
End-to-end ML workflow
Clean preprocessing pipeline
Business-driven decision making
Production-aware design
Easy to explain in interviews

📌 Key Takeaway :
This project focuses not just on building a model, but on building a reliable, explainable, and scalable machine learning system that solves a real business problem.
Easy to explain in interviews


