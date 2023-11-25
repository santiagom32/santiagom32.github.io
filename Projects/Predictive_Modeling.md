---
layout: default
title: "Predictive_Modeling"
---
# Predicting Modeling Using Multiple Linear Regression

## Objective:

To analyze a medical dataset with 50 variables and 10,000 records, focusing on identifying the key health risk factors influencing the length of initial hospitalization ('Initial Days').

## Research Question Analysis:

The research question targets the influence of various variables on the 'Initial Days' of hospitalization. The goal is to determine which factors significantly impact the duration of a patient's first hospital stay.

## Method Justification:

Multiple linear regression was chosen for its ability to handle multiple independent variables and assess their collective impact on a dependent variable. Python was the language of choice due to its robust libraries like Pandas, NumPy, statsmodels, and plotnine, facilitating efficient data manipulation and analysis.

## Data Preparation:

The process involved data cleaning (removing duplicates/nulls), outlier detection, and transformation of categorical variables into numerical data for regression analysis using label encoding and dummy encoding. The data was carefully prepared, ensuring relevance to the research question.

```pyhon
#check for duplicates
print(md_info.duplicated().value_counts())
print(md_info.columns.duplicated().any())
print(md_info.duplicated().any())
```
```
#check for null values
print(md_info.isnull().sum())
```
```
#check outliers created list with values to boxplot
columns_to_plot = [ 'VitD_levels', 'Doc_visits', 'vitD_supp', 'Initial_days','Age','TotalCharge']

# Create individual boxplots for each column
for column in columns_to_plot:
    sns.boxplot(x=column, data=md_info)
    plt.title(f'Boxplot of {column}')
    plt.show()

#generated the z score and printed zscore to detect outliers
md_info['VitD_levels_zscore']=stats.zscore(md_info['VitD_levels'])
print(md_info[['VitD_levels','VitD_levels_zscore']].head)
#generated histogram to detect outliers
plt.hist(md_info['VitD_levels_zscore'])
plt.xlabel('VitD_levels')
plt.ylabel('Frequency')
plt.title('VitD_levels zscore')
plt.show()


#generated the z score and printed zscore to detect outliers
md_info['vitD_supp_zscore']=stats.zscore(md_info['vitD_supp'])
print(md_info[['vitD_supp_zscore','vitD_supp_zscore']].head)
#generated histogram to detect outliers
plt.hist(md_info['vitD_supp_zscore'])
plt.xlabel('vitD_supp')
plt.ylabel('Frequency')
plt.title('vitD_supp zscore')
plt.show()

```
```
#descriptive statistics for all variables
pd.set_option('display.max_columns', None)

md_info_describe = md_info.describe(include='all')

# Display the entire summary statistics
print(md_info_describe)
```

- <img width="418" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/02d2511e-969a-4185-9f1c-da034e48c524">

```
#created a list of variables to plot
columns_to_plot = [ 'Initial_days','Age','ReAdmis','TotalCharge','VitD_levels', 'Doc_visits',
                   'vitD_supp','HighBlood', 'Soft_drink', 'Stroke', 'Complication_risk',
                   'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia',
                   'BackPain','Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis',
                   'Asthma']


# Create the subplot grid 
fig, axes = plt.subplots(4, 5, figsize=(25,15), constrained_layout=True)

#Flatten the axes array to make it easier to iterate through
axes = axes.flatten()

# Loop through each column name and create univariable visualizations
for i, column in enumerate(columns_to_plot):
    sns.histplot(x=column, data=md_info, ax=axes[i])
    axes[i].set_title({column})

# Show the plot
plt.show()

```

<img width="600" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/e3b42c1c-e85e-4e01-a6b4-2aeb857f27be">

```python

# List of explanatory variables names to plot against 'Initial_days'
columns_to_plot = ['Age','TotalCharge','ReAdmis','VitD_levels', 'Doc_visits',
                   'vitD_supp','HighBlood', 'Soft_drink', 'Stroke', 'Complication_risk',
                   'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia',
                   'BackPain','Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis',
                   'Asthma']

# Create the subplot grid
fig, axes = plt.subplots(4, 5, figsize=(25, 15), constrained_layout=True)

# Flatten the axes array to make it easier to iterate through
axes = axes.flatten()

# Loop through each column name and create a scatter plot against 'Initial_days'
for i, column in enumerate(columns_to_plot):
    sns.scatterplot(x=column, y='Initial_days', data=md_info, ax=axes[i])
    axes[i].set_title(f'{column} vs Initial_days')

# Show the plot
```

<img width="600" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/c4268dcd-74db-440c-bea4-bdd8d3ec2a7a">


```
#created dummie variables for nominal categorical values and dropped 1 column 
md_info = pd.get_dummies(data=md_info, columns=['ReAdmis','Soft_drink','HighBlood','Stroke',
                     'Overweight','Arthritis','Diabetes',
                     'Hyperlipidemia','BackPain','Anxiety',
                     'Allergic_rhinitis','Reflux_esophagitis','Asthma'],drop_first=True)
#labelencodign
md_info['Complication_risk'].replace(['Low', 'Medium', 'High'], [1,2,3], inplace =True)
```

## Model Analysis:

The initial model included all potential predictors, which was then refined using backward elimination to focus on statistically significant variables. This approach led to a reduced model that maintained a high R-squared value, indicating the model's effectiveness in explaining the variance of the target variable.

```
#created y with all explanatory variables and x with the target variable
cols = md_info.drop('Initial_days',axis=1).columns.tolist()
x = md_info.drop('Initial_days',axis=1)
y = md_info['Initial_days']
#add intercept
x = sm.add_constant(x)

```
```
#created linear regression model
model=sm.OLS(y,x).fit()
model.summary()
```
<img width="422" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/6fa5040b-b003-4144-9728-17efd7d199d9">

```
#Manually proceeded with backward step elimination until final model:
selected_feature_indices = [0, 2, 3, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18]
x_opt = x.iloc[:, selected_feature_indices]
reduced_model = sm.OLS(endog=y, exog=x_opt).fit()
summary = reduced_model.summary()
print(summary

```
<img width="618" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/39c2d318-a5f0-4d12-8262-22370fbb810b">

```
msr_reduced = reduced_model.mse_resid
print('reduced model Mean Squared Error:',msr_reduced)
rse_reduced = np.sqrt(msr_reduced)
print ('reduced model residual standat error:', rse_reduced)
#diference betwwen predicted initial days and
#observed initial days is about 3.3 days
```
<img width="385" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/59fd9f6c-1cb0-44d6-8bff-4c3c6b94d11e">

```
#residuals vs fitted random scatter
sns.residplot(x=reduced_model.fittedvalues,y=reduced_model.resid, lowess=True)
plt.ylabel('Residuals')
plt.xlabel('Fitted values')
plt.show()
```
<img width="494" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/5ee8601b-eb6c-4e48-8eba-1b2c2327103f">

```
#qq plot 
sm.qqplot(reduced_model.resid, line='45', fit=True)
plt.title('Q-Q plot of residuals')
plt.show()

```
<img width="479" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/128fa5e8-e024-4ace-8172-69189d72afc1">


## Findings and Implications:

The analysis revealed that while the model was statistically significant, the strongest relationships were found with non-health-related variables, suggesting the need for more standardized health statistics for predictive accuracy.
The limitations of the analysis were acknowledged, including assumptions of linearity and multicollinearity, as well as the self-reported nature of the data.

## Recommendations:

Recommended refining the dataset to include more standardized health statistics for better predictability of hospitalization duration based on health-related variables.

- project, I showcased a wide array of data science skills, including robust statistical analysis using multiple linear regression, comprehensive data preparation and cleaning, and adept use of Python and its key libraries like Pandas and NumPy. The project highlighted my proficiency in encoding categorical data for regression, understanding and applying regression assumptions, and employing feature selection techniques like backward elimination for model optimization. Additionally, it demonstrated my ability to interpret complex model outputs, effectively communicate findings, and acknowledge the limitations of the analysis. These skills collectively underscore my capability to tackle real-world data challenges, making this project a significant addition to my professional portfolio in data science.

