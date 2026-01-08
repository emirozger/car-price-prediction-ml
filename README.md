# 🚗 Used Car Price Prediction: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)

This project provides a comprehensive solution for predicting used car prices. It covers the entire data science lifecycle: from raw data cleaning and sophisticated feature engineering to hyperparameter optimization using **Optuna** and a final web deployment via **Streamlit**.

## 📑 Table of Contents
* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Tech Stack](#tech-stack)
* [Data Pipeline](#data-pipeline)
* [Model Performance](#model-performance)
* [Installation & Usage](#installation--usage)
* [Author](#author)

---

## 🎯 Project Overview
The goal of this project is to build a robust regression model that estimates the market value of used cars based on various technical and historical features. The pipeline handles messy real-world data (string symbols, missing values) and uses advanced ensemble methods to achieve high accuracy.



## ✨ Key Features
* **Automated Data Cleaning:** Cleans currency symbols, mileage units, and standardizes categorical text.
* **Regex Feature Extraction:** Extracts `Horsepower`, `Displacement`, and engine configurations (V-engine, Turbo, DOHC) from complex raw text strings.
* **Advanced Feature Engineering:** 
    * `car_age`: Calculated based on the 2026 market year.
    * `Mileage_per_Year`: Normalized usage metric.
    * `hp_per_liter`: A performance efficiency indicator.
    * `is_luxury_brand`: Domain-specific categorical encoding.
* **Outlier Management:** Uses the IQR (Interquartile Range) method to ensure model stability against extreme price and mileage values.
* **Hyperparameter Tuning:** Implementation of **Optuna** for Bayesian optimization of XGBoost parameters.
* **Interactive UI:** A user-friendly Streamlit dashboard for real-time price estimation.

## 🛠 Tech Stack
* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Optimization:** Optuna
* **Deployment:** Streamlit
* **Model Persistence:** Joblib

---

## ⚙️ Data Pipeline

### 1. Preprocessing & Cleaning
* Converted price and mileage from strings (e.g., "$50,000", "10,000 mi.") to numeric formats.
* Imputed missing values using brand-specific modes and medians to maintain data distribution integrity.

### 2. Feature Engineering & Selection
* **Regex Engine:** Parsed the `engine` column to create binary flags for `turbo` and `is_v_engine`.
* **Binning:** Categorized car age into "Age Buckets" to capture non-linear depreciation.
* **Dimensionality Reduction:** Filtered the dataset to focus on the Top 30 most frequent car brands for better generalization.



### 3. Model Training & Optimization
Three models were evaluated using a Pipeline approach:
1. **Linear Regression:** Baseline model with One-Hot Encoding and Scaling.
2. **Random Forest:** Non-linear ensemble approach.
3. **XGBoost:** Gradient boosting optimized via **Optuna** to minimize Mean Absolute Error (MAE).

---

## 📊 Model Performance

| Model | R² Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- |
| Linear Regression | 0.736855 | $7523.004268 |
| Random Forest | 0.814103 | $5954.397656 |
| **XGBoost (Optimized)** | **0.847879** | **$5452.748047** |

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/used-car-price-prediction.git](https://github.com/yourusername/used-car-price-prediction.git)
cd used-car-price-prediction
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Training Pipeline
```bash
python main.py
```
### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
