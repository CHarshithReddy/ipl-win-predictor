# IPL Win Probability Predictor

A ball-by-ball win probability predictor for IPL matches using Machine Learning.

## Live Demo
[Live Demo](https://ipl-win-predit.streamlit.app/)

## What it does
Predicts win probability for every ball in an IPL 2nd innings chase
using XGBoost trained on 125,000+ deliveries across 17 seasons (2008-2024).

## Key Results
| Model | Accuracy | ROC-AUC | Log Loss |
|-------|----------|---------|----------|
| Logistic Regression | 82.3% | 0.905 | 0.392 |
| XGBoost | 79.6% | 0.884 | 0.427 |

## Key Findings
- Chasing teams win 54% of IPL matches overall
- 2016 had highest chasing win rate (68%)
- Wickets in hand after over 15 is the strongest predictor of match outcome
- Match phase (powerplay/middle/death) has surprisingly low predictive impact

## Tech Stack
Python · Pandas · XGBoost · SHAP · Streamlit · Plotly · Scikit-learn

## Dataset
[IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
— 125,741 balls across 1,092 matches (2008-2024)

## Project Structure
notebooks/01_eda.ipynb        → Exploratory data analysis
notebooks/02_features.ipynb   → Feature engineering
notebooks/03_model.ipynb      → Model training & evaluation
app.py                        → Streamlit dashboard

## How to run locally
git clone https://github.com/CHarshithReddy/ipl-win-predictor.git
cd ipl-win-predictor
pip install -r requirements.txt
streamlit run app.py
