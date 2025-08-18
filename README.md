# Fantasy Football AI – Predictive Modeling and Squad Optimization
## Table of Contents
1. [Overview](#overview)  
2. [Methodology](#methodology)  
   - [1. Data Collection](#1-data-collection)  
   - [2. Feature Engineering](#2-feature-engineering)  
   - [3. Model Training](#3-model-training)  
   - [4. Model Evaluation](#4-model-evaluation)  
   - [5. Squad Optimisation](#5-squad-optimisation)  
   - [6. Visualisation & Deployment](#6-visualisation--deployment)  
3. [Models Used and Rationale](#models-used-and-rationale)  
4. [Optimisation Algorithm](#optimisation-algorithm)  
   - [Primary Method: ILP](#primary-method-integer-linear-programming-ilp)  
   - [Fallback Method: Greedy Heuristic](#fallback-method-greedy-heuristic)  
5. [Squad Constraints](#squad-constraints)  
6. [Model Performance Visualisations & Interpretations](#model-performance-visualisations--interpretations)  
   - [1. Actual vs Predicted (Training Data)](#1-actual-vs-predicted-training-data)  
   - [2. Error & Metric Comparison Across Models](#2-error--metric-comparison-across-models)  
   - [3. Predicted Points for the Next Gameweek](#3-predicted-points-for-the-next-gameweek)  
   - [4. Radar Plot (Normalised Metrics)](#4-radar-plot-normalised-metrics)  
7. [Evaluation Summary — Best Model per Metric](#evaluation-summary--best-model-per-metric)  
8. [Conclusions](#conclusions)  
9. [Project Structure](#project-structure)  
10. [Requirements & Installation](#requirements--installation)  
11. [How to Run](#how-to-run)  
12. [References](#references)  

## Overview
**Fantasy Football AI** is an advanced predictive analytics and optimisation system for building high-performing Fantasy Premier League (FPL) squads.  
It uses historical player data and machine learning models to **forecast player points** for upcoming gameweeks, then applies **squad optimisation algorithms** to select the best possible team within FPL’s rules and budget constraints.  

The system is delivered as an **interactive Streamlit web application**, enabling users to:  
- Train and compare multiple ML models on past FPL data  
- Visualise performance metrics and feature importances  
- Optimise squads using Integer Linear Programming (ILP) or a robust greedy fallback  
- Analyse predictions through diagnostics, calibration, and residual checks  
- Export results including squad lists, performance tables, and plots for further analysis  

## Methodology
### 1. Data Collection
- **Source:** [Vaastav FPL Dataset](https://github.com/vaastav/Fantasy-Premier-League)  
- Historical Fantasy Premier League player data including match-by-match statistics, player metadata, and gameweek identifiers.  

### 2. Feature Engineering
- Extracted and cleaned relevant numerical and categorical features:  
  - **Performance metrics:** minutes, goals, assists, clean sheets, bonus points, ICT index.  
  - **Advanced stats:** expected goals (xG), expected assists (xA), expected goal involvements, expected goals conceded.  
- Steps: handle missing values, encode categorical variables, standardise numeric features, aggregate by player & gameweek.  

### 3. Model Training
- Models: Linear Regression (LR), Random Forest (RF), Gradient Boosting (GBR), XGBoost (XGB), KNN  
- Training: all gameweeks except the latest, holding out next GW for prediction.  
- Evaluation: in-sample metrics and optional GroupKFold cross-validation.  

### 4. Model Evaluation
- Metrics: MAE, RMSE, R², Explained Variance, MedAE, Spearman correlation, Top-K overlap.  

### 5. Squad Optimisation
- **Objective:** select an optimal 15-player squad and XI under FPL rules.  
- **Constraints:** budget (100), 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs, max 3 per club, min spend 95%, minutes threshold.  

### 6. Visualisation & Deployment
- Visual outputs: radar charts, residuals, calibration plots, Top-K tables.  
- Deployment: interactive Streamlit web app with exportable results.  

## Models Used and Rationale
1. **Linear Regression** – baseline for interpretability.  
2. **KNN** – non-parametric, predictions based on similarity.  
3. **Random Forest** – ensemble of decision trees, handles non-linearities.  
4. **XGBoost** – high-performance gradient boosting.  
5. **Gradient Boosting Regressor** – sequential boosting, strong on tabular data.  
6. **Ensemble (RF + XGB + GBR)** – combines strengths of tree-based models.  

## Optimisation Algorithm
### Primary Method: Integer Linear Programming (ILP)
- **Library:** PuLP  
- **Objective:** maximise predicted points under FPL constraints.  

### Fallback Method: Greedy Heuristic
- When ILP infeasible or unavailable.  
- Sorts players by predicted points, selects under constraints, builds a valid XI.  

## Squad Constraints
- Squad: 15 players  
- Positions: 2 GK, 5 DEF, 5 MID, 3 FWD  
- Budget: 100 (95% min spend)  
- Max 3 players per club  
- XI formation rules: 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD  
- Captain & Vice-Captain auto-selection  

## Model Performance Visualisations & Interpretations
### 1. Actual vs Predicted (Training Data)
![Training: Actual vs Predicted](plots/ActualvsPredicted.png)  
**Interpretation:** Orange dashed = ideal line. Predictions cluster well for mid-range points. Tree-based models + Ensemble are closest to ideal, LR and KNN weaker.  

### 2. Error & Metric Comparison Across Models
![Model Comparison: MAE, RMSE, R², EV, MedAE, Spearman, TopKOverlap](plots/ErrorComparison.png)  
**Interpretation:** XGBoost and Ensemble best in error (MAE, RMSE) and variance explained. Ranking (Spearman, Top-K overlap) also dominated by tree-based models. LR and KNN underfit the data.  

### 3. Predicted Points for the Next Gameweek
![Predicted Points — Next GW](plots/PredPoints_nxtweek.png)  
**Interpretation:** Forecasted average points per model for GW21. All models predict close values, boosting slightly higher. XGB/Ensemble most reliable for optimisation.  

### 4. Radar Plot (Normalised Metrics)
![Normalised Radar Plot](plots/RadarChart.png)  
**Interpretation:** Normalised (1.0 = best). Ensemble broad coverage across metrics. XGB and RF also strong. LR and KNN show weakest coverage.  

## Evaluation Summary — Best Model per Metric
| Metric         | Best Model    | Score |
|----------------|---------------|-------|
| MAE            | XGBoost       | 0.0287 |
| RMSE           | XGBoost       | 0.0687 |
| R²             | XGBoost       | 0.9992 |
| Explained Var. | XGBoost       | 0.9992 |
| MedAE          | Random Forest | 0.0000 |
| Spearman       | Random Forest | 0.9792 |
| Top-K Overlap  | Ensemble      | 0.9000 |
**Interpretation:** XGBoost best in accuracy metrics. Random Forest strongest in consistency and correlation. Ensemble excels in ranking ability (Top-K overlap).  

## Conclusions
Ensemble and XGBoost consistently outperform simpler models. Tree-based methods handle non-linearity and ranking better. Best pipeline = **Ensemble/XGB predictions → ILP squad optimisation**.  

## Project Structure
```
project/
├── app.py # Streamlit application
├── README.md # Documentation
├── data/ # Input datasets
│ └── merged_gw.csv
├── plots/ # Generated plots and visualisations
│ ├── ActualvsPredicted.png
│ ├── ErrorComparison.png
│ ├── PredPoints_nxtweek.png
│ ├── RadarChart.png
│ ├── En_Histogram.png
│ ├── GB_Histogram.png
│ ├── kNN_Histogram.png
│ ├── LR_Histogram.png
│ ├── RF_Histogram.png
│ ├── XG_Histogram.png
│ ├── En_callibration.png
│ ├── GB_callibration.png
│ ├── kNN_callibration.png
│ ├── LR_callibration.png
│ ├── RF_callibration.png
│ ├── XG_callibration.png
│ ├── PPM_MAE.png
│ ├── PPM_RMSE.png
│ └── ... (additional residuals and metrics)
├── tables/ # Evaluation tables
│ ├── ModelComparison.csv
│ ├── PerPositionMetric.csv
│ └── Squad_XGBoost.csv
├── PROJECT_POSTER.pdf # Final poster
└── Literature_Review.pdf # Supporting reference material
```

## Requirements & Installation
Ensure you have **Python 3.9+** installed.  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

## How to Run

Clone or download this repository.

1. Navigate into the project folder in terminal or VS Code.
2. Place your input dataset (merged_gw.csv) into the data/ directory.
3. Install dependencies:
pip install -r requirements.txt
4. Run the Streamlit app:
streamlit run app.py

5. Use the app to:
Train models and compare performance
View plots and error metrics
Optimise a squad using ILP or greedy fallback

## References
[1] Vaastav Anand. Fantasy Premier League Data Repository. https://github.com/vaastav/Fantasy-Premier-League, 2024. Accessed: 2025-06-13.  
[2] Malhar Bangdiwala, Rutvik Choudhari, Adwait Hegde, and Abhijeet Salunke. Using ML models to predict points in Fantasy Premier League. In *2022 2nd Asian Conference on Innovation in Technology (ASIANCON)*, pages 1–6. IEEE, 2022.  
[3] IBM. IBM Watson Fantasy Football. https://www.ibm.com/thought-leadership/fantasy-football/index.html, 2024. Accessed: 2025-06-13.  
[4] Prof. Dr. S. N. Sarda, Rishikesh Sahu, Atharva Ingole, Aditi Patil, and Akansha Tarpe. Fantasy Football Team Prediction Using ML. https://www.doi.org/10.56726/IRJMETS54188, 2024. Accessed: 2025-06-13.  
[5] Mitchell, S., Mason, A., & OSI Community. PuLP: A Linear Programming Toolkit for Python. https://coin-or.github.io/pulp/  
[6] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.  
[7] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.  
[8] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *KDD '16*.  
[9] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.  