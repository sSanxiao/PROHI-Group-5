# PROHI Sepsis Prediction Dashboard

## Overview
This project provides a comprehensive dashboard for sepsis prediction and analysis in a clinical setting. The dashboard uses machine learning to predict sepsis risk and provides tools for understanding and interpreting these predictions.

## Features
The dashboard consists of five main components:

1. **Descriptive Analytics**: Explore and understand the dataset through visualizations and summaries.
2. **Diagnostic Analytics**: Analyze relationships between variables and identify factors correlated with sepsis.
3. **Predictive Analytics**: Generate sepsis risk predictions for individual patients using a pre-trained Random Forest model.
4. **Prescriptive Analytics**: Understand model predictions through SHAP values and get clinical recommendations.
5. **About**: Information about the project, dataset, team members, and references.

## Dataset
The project uses the [Prediction of Sepsis dataset from Kaggle](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis), which contains patient measurements from ICU settings.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/PROHI-Group-5/sepsis-prediction.git
   cd sepsis-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data` directory as `cleaned_dataset.csv`.

4. Run the dashboard:
   ```
   streamlit run Dashboard.py
   ```

## Project Structure
- `Dashboard.py`: Main application file that integrates all components
- `Descriptive_Analytics.py`: Code for the descriptive analytics tab
- `Diagnostic_Analytics.py`: Code for the diagnostic analytics tab
- `Predictive_Analytics.py`: Code for the predictive analytics tab
- `Prescriptive_Analytics.py`: Code for the prescriptive analytics tab
- `About.py`: Information about the project and team
- `data/`: Directory for dataset files
- `models/`: Directory for trained machine learning models
- `assets/`: Directory for images and other assets

## Team Members
- Max Altez Linhardt
- Khachatur Dallakyan
- Pratibha Rustogi
- Qilu Wang
- Xue Wu

## References
1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801–810.
2. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine. 2020;48(2):210-217.
3. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems. 2017;30:4765-4774.
4. Kaggle Dataset: [Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

## License
This project is for educational purposes only.