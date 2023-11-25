---
layout: default
title: "Predicting modeling using Logistic Regression"
---

# Predicting Modeling Using Logistic Regression

## Objective:

This project aimed to apply logistic regression to a comprehensive medical dataset, with the goal of identifying key variables that predict the probability of patient readmission. This analysis is vital for healthcare organizations seeking to minimize readmission rates and enhance patient care.

## Research Question Analysis:

The central research question focused on determining the factors within a medical dataset of 50 variables and 10,000 records that influence the likelihood of a patient being readmitted. The dataset's complexity offered a rich opportunity to uncover significant health risk factors.

## Method Justification:

Logistic regression was the chosen method due to the binary nature ('Yes' or 'No') of the target variable, 'Readmis'. Python's extensive data processing capabilities made it an ideal choice for handling the dataset. The use of Python, favored for its adaptability in various industries and powerful processing capacity, allowed for efficient manipulation, processing, and analysis of the extensive data.

## Data Preparation:

The data preparation process was comprehensive, ensuring the dataset's readiness for logistic regression. This involved meticulous cleaning steps such as the removal of duplicates, null values, and irrelevant columns, as well as outlier treatment. Key variables like 'ReAdmis', 'Complication_risk', and 'Initial_days' were transformed from categorical to numerical forms using encoding techniques.

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
c#drop non relevant variables
md_info_drop = ["CaseOrder","Customer_id","Interaction","UID","City","State",'Doc_visits','Full_meals_eaten',
                     "County","Zip","Lat","Lng","Population","Area","TimeZone",'Initial_admin','Soft_drink',
                     "Job","Children","Income","Marital","Additional_charges","Item1",
                     "Item2","Item3","Item4","Item5","Item6","Item7","Item8",'Gender','TotalCharge']
md_info.drop(columns=md_info_drop, inplace=True)
#set target variable to be the first column in dataset

```
```
#check outliers created list with values to boxplot
columns_to_plot = [ 'Age','vitD_supp','VitD_levels', 'Initial_days']

# Create individual boxplots for each column
for column in columns_to_plot:
    sns.boxplot(x=column, data=md_info)
    plt.title(f'Boxplot of {column}')
    plt.show()
```

<img width="187" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/c663b5c5-33ac-4724-84ac-1244ecc3068c">

```
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

<img width="519" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/0573a4e8-7930-4e9a-8441-ecfc54eb56e4">

```
#created dummie variables for nominal categorical values and dropped 1 column 
md_info = pd.get_dummies(data=md_info, columns=['Services','HighBlood','Stroke',
                     'Overweight','Arthritis','Diabetes',
                     'Hyperlipidemia','BackPain','Anxiety',
                     'Allergic_rhinitis','Reflux_esophagitis','Asthma'],drop_first=True)

md_info['Complication_risk'].replace(['Low', 'Medium', 'High'], [1,2,3], inplace =True)
md_info['ReAdmis'].replace(['Yes','No'],[1,0],inplace=True)
```

**exploratory analaysis**
- Univariable Visualizations:

<img width="735" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/d0d124e1-a244-40f1-8705-0ceff4fcc1cf">


- Bivariable Visualizations:

<img width="493" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/3679ef19-0db5-4c8d-aae6-3c4f22228002">

<img width="476" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/f8866bc2-d822-4ed3-87c1-78fbfa3da17b">

<img width="486" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/9920d73e-6dcd-456b-9cef-7a3dff054585">

<img width="735" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/5801a3d2-7d89-42bc-b1d6-5986c86cc68a">



  
## Model Analysis:

An initial logistic regression model incorporating all 19 identified features was constructed and subsequently refined to a reduced model through backward step elimination. This process aimed at eliminating non-significant features, resulting in a model with 14 relevant predictors. The reduced model was compared with the initial one, focusing on accuracy and the interpretation of coefficients.

```
#splited the data 80% into training and 20% into test dataset & added constant to both x train and x test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)
```
```
#created initial model
initial_model=sm.Logit(y_train,x_train_const).fit()
print(initial_model.summary())
```

<img width="656" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/fdb0a8ce-c937-4274-8337-fc5970362d74">

**Check accuracy initial model**
```
#extracted Y predictions from intial model and used function accuracy score to extract it 
y_pred_initial=(initial_model.predict(x_test_const)>0.5).astype(int)
print("inital model accuracy", accuracy_score(y_test,y_pred_initial))
```

<img width="219" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/3a207346-bc56-4e2e-acc7-01357bfcaaae">


<img width="249" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/30fe9a22-23ce-4f97-b5a3-e5b4df9da0f5">


<img width="567" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/506b481a-355a-4bfd-9493-9708efbc2f6b">

**backward elimination to get reduced model**

```
x_train_opt = x_train_const.copy()

while True:
    # Fit the model with current features
    model = sm.Logit(y_train,x_train_opt).fit()
    
    # Find the feature with the highest p-value
    max_p_value = max(model.pvalues)
    max_p_feature = model.pvalues.idxmax()

    # If the max p-value is greater than 0.05, drop the feature
    if max_p_value > 0.05:
        x_train_opt = x_train_opt.drop(columns=[max_p_feature])
    else:
        break
        
#generated reduced model
Reduced_model = sm.Logit(endog=y_train, exog=x_train_opt).fit()
Reduced_model.summary()
```
<img width="401" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/11a491b0-54d0-46d7-8a35-2b11eaf42753">

**Reduced model accuracy**

```
#extracted Y predictions from reduced model and used function accuracy score to extract it 
y_pred_reduced=(Reduced_model.predict(x_test_opt)>0.5).astype(int)
print("reduced model accuracy", accuracy_score(y_test,y_pred_reduced))
```
<img width="234" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/d3fd7df3-f60a-4007-8dd1-028fa70ee6b3">

```
#created a reduced confusion matrix with actual data and predicted data from previous step
reduced_confusion_matrix=confusion_matrix(y_test,y_pred_reduced)
contingency_table_reducedcm = pd.crosstab(y_test, y_pred_reduced, rownames=['Actual'], colnames=['Predicted'])

#created heat map to visualize reduced model data
plt.figure(figsize=(8,6))
sns.heatmap(reduced_confusion_matrix, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])

plt.ylabel('actual')
plt.xlabel('Predicted')
plt.title('reduced model Confusion Matrix ')
plt.show()
```

<img width="578" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/b8265dec-7289-43e8-be04-9d5b1f0f3c0e">

```
#identified True negatives, true positves, false negatives and false positives

TN=reduced_confusion_matrix[0,0]
TP=reduced_confusion_matrix[1,1]
FN=reduced_confusion_matrix[1,0]
FP=reduced_confusion_matrix[0,1]
#calculated accuracy, sensitivity and specificity
accuracy=(TN+TP)/(TN+TP+FN+FP)
print('reduced model accuracy',accuracy)
Sensitivity=TP/(FN+TP)
print('reduced model sensitivity',round(Sensitivity,4))
Specificity=TN/(TN+FP)
print('reduced model specificity',round(Specificity,4))
```
## Findings and Implications:

The reduced model effectively highlighted the influence of various health-related factors on readmission rates. Key findings included the impact of initial hospitalization duration, complication risks, and specific medical services on readmission likelihood. Despite its high accuracy, the model's potential overfitting and reliance on binary medical features were noted as limitations, suggesting the need for further data enrichment and analysis.

## Recommendations:

for this project is recommended to continuously update the model with new patient data to validate and enhance its predictive accuracy. Further, a more granular examination of statistically significant variables using detailed health data could provide deeper insights into specific risk factors affecting patient readmission.

## Significance:

In this project project, I adeptly applied logistic regression to a medical dataset, showcasing my skills in statistical analysis, data preparation, and Python programming. The project involved cleaning and transforming a large dataset, understanding and applying logistic regression assumptions, and employing feature selection for model optimization. My analysis focused on identifying key factors influencing patient readmission, demonstrating my ability to interpret complex data and extract meaningful insights, crucial for informed decision-making in healthcare. This work highlights my proficiency in data science, particularly in handling real-world data challenges and deriving actionable knowledge from them.
