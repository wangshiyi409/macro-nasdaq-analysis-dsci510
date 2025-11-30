# Macro Indicators and NASDAQ Tail-Risk Analysis  
DSCI 510 – Final Project  
Author: Shiyi Wang (shiyiw@usc.edu)  
USC ID: 9862305589

---

## 📌 Project Overview

This project investigates the relationship between major U.S. macroeconomic indicators and the NASDAQ index, with a focus on understanding:

1. **Correlation** between macro indicators and equity market performance  
2. **Predictive ability** of macroeconomic factors for NASDAQ tail-risk events  
3. **Statistical & visualization-driven insights** into market behavior under macro conditions

Data is collected programmatically from the **FRED API** and **Yahoo Finance API**, cleaned, merged, and analyzed using Python.

This repository follows the required structure specified in the DSCI 510 Final Project guidelines.

---

## 📁 Repository Structure
```text
.
├── README.md
├── requirements.txt
├── project_proposal.pdf
├── data/
│ ├── raw/ # Raw data fetched from APIs
│ └── processed/ # Cleaned & merged datasets
├── results/
│ ├── final_report.pdf # Final report 
│ ├── macro_timeseries.png
│ ├── correlation_heatmap.png
│ ├── roc_curve.png
│ ├── confusion_matrix.png
│ └── macro_nasdaq_analysis.ipynb
└── src/
  ├── get_data.py
  ├── clean_data.py
  ├── run_analysis.py
  ├── visualize_results.py
  └── utils/
    ├── fred_api.py
    ├── yahoo_api.py
    └── helpers.py
```
---

## ⚙️ Installation Instructions

### 1️⃣ Create and activate a virtual environment

python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows

shell
复制代码

### 2️⃣ Install all required dependencies

pip install -r requirements.txt


This will install packages such as:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- requests  
- yfinance  

---

## 📥 Step 1 — Data Collection

The script downloads macroeconomic indicators from **FRED API** and NASDAQ price data from **Yahoo Finance**.  
Raw files are automatically saved under `data/raw/`.

Run:

python src/get_data.py


This script will:

- Fetch GDP, CPI, UNRATE, FEDFUNDS, DGS3MO, VIX, etc.
- Fetch NASDAQ daily close prices
- Save raw output as `.csv` or `.json`

---

## 🧹 Step 2 — Data Cleaning & Processing

This step merges different data sources, aligns time indices, converts frequencies, and creates the final feature DataFrame.

Run:

python src/clean_data.py


Output is stored in:

data/processed/merged_macro_nasdaq.csv


---

## 📊 Step 3 — Analysis & Modeling

This script performs:

- Correlation analysis  
- Tail-risk label generation  
- Logistic regression model training  
- Evaluation (AUC, precision, recall, confusion matrix)

Run:

python src/run_analysis.py


Results are printed to console and saved to `/results/`.

---

## 📈 Step 4 — Visualization

Generate all plots used in the final report:

- Macro indicator time series
- Correlation heatmap
- ROC curve
- Confusion matrix

Run:

python src/visualize_results.py


Plots will be saved under:

results/


---

## 📝 Final Report

The **final_report.pdf** (2–5 pages) summarizes:

- Motivation and research question  
- Data collection and API sources  
- Cleaning and analysis methods  
- Visualizations and interpretation  
- Changes from original proposal  
- Future work  

This file is located under:

results/final_report.pdf


---

## 🚀 How to Reproduce the Entire Pipeline

To reproduce the full workflow from raw data to final figures:

python src/get_data.py
python src/clean_data.py
python src/run_analysis.py
python src/visualize_results.py

---

## 📚 Data Sources

- Federal Reserve Economic Data (FRED): https://fred.stlouisfed.org  
- Yahoo Finance API (via yfinance): https://pypi.org/project/yfinance/  

---

## ✔️ Notes

- All raw and processed data included are <100MB (required by GitHub).  
- Jupyter Notebook is used **only for visualization** as required by the project rubric.  
- Core logic resides in `.py` modules under `/src/`.

---

## 📧 Contact

If you have questions regarding this project:  
**Shiyi Wang**  
Email: **shiyiw@usc.edu**
