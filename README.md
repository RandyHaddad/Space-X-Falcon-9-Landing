# SpaceX Falcon 9 First Stage Landing Prediction

[**👉 Jump to Full Prediction Notebook**](https://github.com/RandyHaddad/Space-X-Falcon-9-Landing/blob/master/Space-X-Falcon-9-Landing-Prediction.ipynb)

Thank you to IBM for support and resources that made this project possible.

## Project Overview

SpaceX Falcon 9 rockets can be reused, reducing the cost of launches significantly. This project uses machine learning to predict if the first stage of a Falcon 9 rocket will land successfully, providing insights into cost predictions and enabling potential competitive bids against SpaceX launches.

## Objectives

- Perform exploratory data analysis to generate insights
- Create labels for successful landings
- Standardize and split data for training and testing
- Evaluate models and tune hyperparameters (SVM, Decision Trees, Logistic Regression, and KNN)

## First Stage Landing Prediction

In this section, we build a machine learning pipeline to predict Falcon 9 first stage landings based on historical data. Our pipeline includes data preprocessing, training with different algorithms, and model evaluation.

### Models Used
1. **Logistic Regression** – Best hyperparameters found using GridSearchCV
2. **Support Vector Machine (SVM)** – Tested with multiple kernel types and gamma values
3. **Decision Tree** – Tuned for optimal splitting and depth
4. **K-Nearest Neighbors (KNN)** – Experimented with various neighbor counts and distance metrics

### Implementation
Using a variety of machine learning models, we compared performance and selected the optimal parameters for each model to achieve maximum accuracy.

### Accuracy Results
All models performed consistently, achieving an accuracy of approximately 83% on test data:
- **Logistic Regression**: 83.3%
- **SVM**: 83.3%
- **Decision Tree**: 83.3%
- **KNN**: 83.3%

## Setup and Installation

### Prerequisites
Ensure `Python` and libraries such as `NumPy`, `Pandas`, `Seaborn`, and `scikit-learn` are installed. You can install these with:
```bash
pip install numpy pandas seaborn scikit-learn
