---
layout: default
title: "Predictive_Modeling"
---
# Exploratory Data Analysis Using Multiple Linear Regression

## Objective:

To analyze a medical dataset with 50 variables and 10,000 records, focusing on identifying the key health risk factors influencing the length of initial hospitalization ('Initial Days').

## Research Question Analysis:

The research question targets the influence of various variables on the 'Initial Days' of hospitalization. The goal is to determine which factors significantly impact the duration of a patient's first hospital stay.

## Method Justification:

Multiple linear regression was chosen for its ability to handle multiple independent variables and assess their collective impact on a dependent variable. Python was the language of choice due to its robust libraries like Pandas, NumPy, statsmodels, and plotnine, facilitating efficient data manipulation and analysis.

## Data Preparation:

The process involved data cleaning (removing duplicates/nulls), outlier detection, and transformation of categorical variables into numerical data for regression analysis. The data was carefully prepared, ensuring relevance to the research question.

## Model Analysis:

The initial model included all potential predictors, which was then refined using backward elimination to focus on statistically significant variables. This approach led to a reduced model that maintained a high R-squared value, indicating the model's effectiveness in explaining the variance of the target variable.

## Findings and Implications:

The analysis revealed that while the model was statistically significant, the strongest relationships were found with non-health-related variables, suggesting the need for more standardized health statistics for predictive accuracy.
The limitations of the analysis were acknowledged, including assumptions of linearity and multicollinearity, as well as the self-reported nature of the data.

## Recommendations:

Recommended refining the dataset to include more standardized health statistics for better predictability of hospitalization duration based on health-related variables.
