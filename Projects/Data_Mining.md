---
layout: default
title: "Data Mining KNN "
---

- Objective:

To apply the K-nearest neighbor (KNN) classification method to a medical dataset, aiming to predict patient readmission rates based on various health-related variables.

- Research Question Analysis:

The project centered around determining if patient readmission can be predicted using other variables in the dataset. This analysis is crucial for healthcare organizations looking to reduce readmission rates.

- Method Justification:

KNN was chosen for its ability to classify binary outcomes effectively, making it suitable for predicting readmission ('Yes' or 'No'). The method leverages the proximity of data points to predict outcomes, aligning with the categorical nature of the target variable.

- Data Preparation:

The data preparation process involved thorough cleaning, including handling duplicates, null values, and outliers. A range of both continuous (like 'Age' and 'VitD_levels') and categorical variables (like 'HighBlood' and 'Stroke') were selected for the analysis, with one-hot encoding applied to categorical variables for effective processing.

- Analysis Technique:

The KNN model was trained using a split dataset approach, optimizing for the best n_neighbors value using GridSearchCV. The model's performance was evaluated using accuracy scores, ROC curve analysis, and confusion matrices.

- Findings and Implications:

The model achieved a high accuracy score and AUC, indicating its effectiveness in predicting readmissions. However, a limitation was noted in its inability to quantify the influence of individual variables on the prediction. The results suggest that while the model is reliable in classifying readmissions, further refinement and analysis could enhance its predictive capabilities.

- Recommendations:

Future actions include testing the model with new data to validate its predictive power and refining data collection processes for medical variables to increase analysis accuracy. Additionally, conducting feature selection on explanatory variables could provide deeper insights into their relationships with readmissions.
