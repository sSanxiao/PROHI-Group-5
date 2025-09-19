# PROHI Sepsis Prediction Dashboard

**Team Project - PROHI Course - Group 5**

**Team Members:**
- Max Altez Linhardt
- Khachatur Dallakyan  
- Pratibha Rustogi
- Qilu Wang
- Xue Wu

![Dashboard Logo](./assets/project-logo.jpg)

## Project Overview

This dashboard demonstrates early sepsis prediction for ICU patients using machine learning techniques. The project analyzes a comprehensive dataset with 44 clinical features including vital signs, lab values, and demographics to predict sepsis onset.

## Dataset Setup

**Important:** You need to download the dataset before running the dashboard.

1. Download the "Prediction of Sepsis" dataset from Kaggle: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
2. Create a `data` folder in the project root directory
3. Place the downloaded `Dataset.csv` file in the `data/` folder

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:
   - **Windows PowerShell:** `.\env\Scripts\Activate.ps1`
   - **Windows Command Prompt:** `.\env\Scripts\activate.bat`
   - **Linux/Mac:** `source env/bin/activate`

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

Make sure your virtual environment is activated, then run:

```bash
streamlit run Dashboard.py
```

If the above command fails, try:
```bash
python -m streamlit run Dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## Dependencies

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Pillow (PIL)
