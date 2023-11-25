Project Title: Predictive Modeling – D208 Task II

Objective:

To analyze a medical dataset using logistic regression to identify factors predicting patient readmission.
Research Question Analysis:

The research question seeks to determine which variables in a medical dataset predict the likelihood of a patient being readmitted.
Method Justification:

Logistic regression was chosen due to the categorical nature of the target variable ('Readmis'), which indicates readmission as 'Yes' or 'No'. Python was used for its broad industry application and powerful data processing capabilities.
Data Preparation:

Data cleaning involved removing duplicates and nulls, treating outliers, and transforming categorical variables into numerical formats. Variables like 'ReAdmis', 'Complication_risk', and various health indicators were encoded for analysis.
Model Analysis:

The initial logistic regression model was compared to a reduced model, refined using backward step elimination to focus on statistically significant variables. The reduced model maintained high accuracy in predicting readmissions, with minor changes in prediction rates compared to the initial model.
Findings and Implications:

The analysis highlighted relationships between health factors and readmission rates. The reduced model's coefficients provided insights into how different variables influenced readmission odds, such as the impact of initial hospital stay duration and medical services received. The high accuracy and statistical significance of the model demonstrated its practical utility in identifying readmission risk factors.
Recommendations:

Further analysis with new patient data is recommended to test the model's performance and refine the understanding of specific risk factors. More granular medical data could enhance the accuracy of the predictive analysis.
