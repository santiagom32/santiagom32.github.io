---
layout: default
title: "Data Mining 1"
---

# Health Care Lasso Regression

## Objective:

The project aimed to leverage Lasso regression for analyzing a comprehensive medical dataset to identify key predictors of the initial length of hospital stays ('Initial Days'). This objective was aligned with the broader goal of helping healthcare facilities reduce patient readmission rates by identifying and addressing critical risk factors.

## Research Question Analysis:

The central research question was: "Can the 'Initial Days' variable be predicted by other variables in the medical dataset, and what are the most influential variables?" This question was pivotal in understanding the factors contributing to prolonged initial hospital stays, which is a significant concern in healthcare management.

## Method Justification:

Lasso regression was selected for its proficiency in feature selection and handling datasets with numerous variables. Its capability to apply L1 regularization effectively reduces the impact of less significant variables, thereby spotlighting the most influential factors. This method was expected to yield insights into which variables most strongly correlate with the length of initial hospital stays.

## Data Preparation:

The data preparation phase involved converting all categorical variables into numeric formats and scaling continuous variables, a crucial step for Lasso regression analysis. Techniques such as one-hot encoding and label encoding were used to prepare the dataset, ensuring it was suitable for the regression analysis. The preparation process was meticulous, focusing on maintaining data integrity and relevance to the research question.

```
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
#drop non relevant variables
md_info_drop = ["CaseOrder","Customer_id","Interaction","UID","City","State",
                     "County","Zip","Lat","Lng","Population","Area","TimeZone",
                     "Job","Children","Marital","Initial_admin","Additional_charges","Item1",
                     "Item2","Item3","Item4","Item5","Item6","Item7","Item8",'Gender','Full_meals_eaten','Services']
md_info.drop(columns=md_info_drop, inplace=True)

#set target variable to be the first column in dataset
md_info=md_info[['Initial_days','Age','ReAdmis','TotalCharge','VitD_levels','Doc_visits',
                 'vitD_supp','Soft_drink','HighBlood','Stroke','Complication_risk',
                 'Overweight','Arthritis','Diabetes','Hyperlipidemia','BackPain',
                 'Anxiety','Allergic_rhinitis','Reflux_esophagitis','Asthma']]
md_info.head()
```

```
#check outliers created list with values to boxplot
columns_to_plot = [ 'VitD_levels', 'Doc_visits', 'vitD_supp', 'Initial_days','Age','TotalCharge']

# Create individual boxplots for each column
for column in columns_to_plot:
    sns.boxplot(x=column, data=md_info)
    plt.title(f'Boxplot of {column}')
    plt.show()
```

```
#descriptive statistics for all variables
pd.set_option('display.max_columns', None)

md_info_describe = md_info.describe(include='all')

# Display the entire summary statistics
print(md_info_describe)
```

<img width="487" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/a343668d-e872-4611-bcb5-b4f8230502e4">

```
# Feature Scaling
columnstoscale=['Initial_days','Age','TotalCharge','VitD_levels','Doc_visits','vitD_supp',]
md_info[columnstoscale]=scale(md_info[columnstoscale])
md_info.head()
```
```

#created dummie variables for nominal categorical values and dropped 1 column 
md_info = pd.get_dummies(data=md_info, columns=['ReAdmis','Soft_drink','HighBlood','Stroke',
                     'Overweight','Arthritis','Diabetes',
                     'Hyperlipidemia','BackPain','Anxiety',
                     'Allergic_rhinitis','Reflux_esophagitis','Asthma'],drop_first=True)
#label encofing 
md_info['Complication_risk'].replace(['Low', 'Medium', 'High'], [1,2,3], inplace =True)
```
## Analysis Technique:

Employing Python and its powerful data analysis libraries, including Pandas for data manipulation and Scikit-learn for regression modeling, the Lasso regression model was developed and refined. The model incorporated cross-validation to determine the optimal alpha value, ensuring the selection of the most relevant features influencing 'Initial Days'.

```
#splited the data 80% into training and 20% into test dataset & added constant to both x train and x test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#joined training and testing sets for export
training_set = pd.concat([x_train, y_train], axis=1)
testing_set = pd.concat([x_test, y_test], axis=1) 
#exported train and test sets
training_set.to_csv('/Users/santiago/Desktop/WGU /MS data analytics/D209 - Data Mining I /D209 task 2/tainingset-D209-2.csv', index = False)
testing_set.to_csv('/Users/santiago/Desktop/WGU /MS data analytics/D209 - Data Mining I /D209 task 2/testingset-D209-2.csv', index = False)
#added constant to te x sets for modeling
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)
```
```
#model creation
Lasso_cv = LassoCV(cv=10)
Lasso_cv.fit(x_train,y_train)
best_alpha= Lasso_cv.alpha_
print(round(best_alpha,5))
```
<img width="49" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/faea01b3-b28b-4f8b-a5e4-6ab94332eb0e">

```
Lasso_model= Lasso(alpha=0.00099)
Lasso_model.fit(x_train,y_train)
y_pred =Lasso_model.predict(x_test)
print("R^2: {}".format(Lasso_model.score(x_test,y_test)))
rmse= np.sqrt(mean_squared_error(y_test,y_pred))
print('Mean Squared Error',(mean_squared_error(y_test,y_pred)))
print('Root Mean squared Error: {}'.format(rmse))
```
<img width="303" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/d1fabf92-cc12-4240-97ed-da75f15acf10">

- Model Performance

```
variable_names = md_info.drop('Initial_days', axis=1).columns
lasso_coef= Lasso_model.fit(x,y).coef_
plt.plot(range(len(variable_names)),lasso_coef)
plt.xticks(range(len(variable_names)),variable_names.values,rotation=90)
plt.margins(0.02)
plt.show()
```
<img width="446" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/62e61664-f35d-490d-9f06-cffe3ab1d70c">

```
coefficients = Lasso_model.coef_
col = x_train.columns
fig, ax = plt.subplots()
width = 0.6
ind = np.arange(len(coefficients))
ax.barh(ind, coefficients, width, color='green')
ax.set_yticks(ind + width / 10)
ax.set_yticklabels(col, minor=False)
plt.show()
```
<img width="523" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/da3ff107-cf0d-401a-9d9a-5ae7353a7cfc">

```
variable_names = md_info.drop('Initial_days', axis=1).columns
coefficients = Lasso_model.coef_

print("Intercept:", Lasso_model.intercept_)
print("Coefficients:")
for name, coef in zip(variable_names, coefficients):
    print(f"{name}: {coef}")
```
<img width="288" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/bd09a243-9e9a-45dd-92c5-1e42d249fe3b">

```
#denormalized data with = scaled data + data_std(26.309341) + data_mean(34.455299) 
#this values were extracted form data prep descriptive statistics
y_pred_denormalized = y_pred * 26.309341 + 34.455299
y_test_denormalized = y_test * 26.309341 + 34.455299
plt.scatter(y_test_denormalized, y_pred_denormalized, c="green", label='Predicted Values')

plt.scatter(y_test_denormalized, y_test_denormalized, c="blue", label='True Values')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()

```

<img width="459" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/39a73cde-4342-4ee5-a7ca-fa6b9a74841f">

## Findings and Implications:

The Lasso regression model exhibited high accuracy, as reflected in its R-squared value. Interestingly, the analysis revealed that 'Total Charge' was the predominant significant feature impacting 'Initial Days'. This finding suggested a potential limitation in the dataset's variable diversity, highlighting the need for a more comprehensive range of variables to gain a deeper understanding of the factors affecting hospital stay durations.

## Recommendations:

o enhance the model's utility and accuracy, it is recommended to expand the dataset with more granular and continuous medical variables. This would allow for a more detailed and nuanced analysis, potentially uncovering additional key factors influencing patient hospital stays. Furthermore, the organization should consider exploring different analytical models or incorporating more diverse data sources to broaden the scope of the analysis.
