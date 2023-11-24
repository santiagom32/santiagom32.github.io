---
layout: default
title: "Exploratory Data Analysis"
---
# Exploratory Data Analysis

## Objective:

To explore the relationship between patient overweight status and hospital readmission rates using a comprehensive medical dataset.

## Research Question Analysis:

Investigated whether the 'ReAdmin' variable (readmission within 30 days) is dependent on the 'Overweight' variable in patient records. Analyzing this relationship can help healthcare organizations in identifying overweight as a potential risk factor for readmission.

## Data Identification:

Focused on two primary variables: 'ReAdmin', a categorical variable indicating readmission status, and 'Overweight', a categorical variable based on patient's gender, height, and age. The dataset comprised 10,000 records with these key variables.

## Data Analysis:

Utilized the Chi-square test to analyze the dependence between 'ReAdmin' and 'Overweight'.
Chose this method due to the categorical nature of the variables.
The test's outcome suggested that 'ReAdmin' is independent of 'Overweight'.

```python
#create contingency table
overw_readm=pd.crosstab(md_info['ReAdmis'],md_info['Overweight'])

print(overw_readm)

#perform Chi-squared test test
chi, p, dof, expected = chi2_contingency (overw_readm)
print ("Chi = ", str(chi))
print ("p-value =", str(p))
print ("degrees of freedom=", str(dof))
alpha = 0.05 
if p <= alpha:
    print('dependent, reject null hypothesis')
else:
    print ("independent, null hypothesis is true")
```
<img width="252" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/73134325-716e-4698-9fa9-d9345b8e814f">


## Univariable Statistics:

Analyzed the distribution of continuous variables (patient income and total charges) and categorical variables (gender and state) using histograms and bar charts.
Identified distributions like bimodal and positive skewness in these variables.

<img width="195" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/205fa8a2-a723-483f-8c27-e354aa44af13">

<img width="195" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/1fbd0d16-7b69-4b3f-bb1c-2c8b505026d0">

<img width="195" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/6f58933c-4a5d-42c0-9aae-1d4bbc7d6eb5">

<img width="446" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/5ded328c-5243-4150-92f7-3bf4bc25eb3c">


## Bivariate Statistics:

Examined the correlation between continuous variables (age and income) and the relationship between categorical variables (Overweight and soft drink consumption) using scatter plots and bar charts.
Found a uniform distribution for age vs. income and diverse drinking habits.

<img width="200" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/4d2d666a-f28a-4290-8520-980acae8c0fd">

<img width="200" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/cc6bd99b-402c-446c-91c1-781aafd4c381">


## Implications and Significance::

The hypothesis test indicated no significant relationship between readmission rates and overweight status.
Noted limitations in the analysis include the focus on just two of fifty variables and potential data collection biases.
Recommended further analysis on other variables to identify more risk factors for readmission.


This project showcased my ability to conduct complex statistical analyses in a healthcare setting, utilizing techniques like Chi-square testing and univariate and bivariate statistics to draw meaningful conclusions from medical data.
