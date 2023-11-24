---
layout: default
title: "Data Cleaning"
---
# Data Cleaning & PCA

In this data analysis project, I focused on integrating and interpreting data from a comprehensive medical dataset. The primary objective was to assess whether age and being overweight are influential factors in hospital readmission rates.

## Research Question Analysis:
The project aimed to determine the impact of age and overweight status on readmission rates, a question pertinent to healthcare organizations.

## Data Identification:
The dataset comprised 50 variables across 10,000 patient records, covering diverse aspects such as patient demographics, health conditions, and treatments. Notable variables included CaseOrder, Customer_id, Interaction, City, State, County, Zip, Lat, Lng, Population, Area, and several health-related factors like VitD_levels, HighBlood, Stroke, and Overweight.

```
#import libraries to phyton
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import missingno as msno
```
```
#import the medical dataset 
md_info=pd.read_csv('/Users/santiago/Desktop/Data_Cleaning_D206/medical data copy/medical_raw_data.csv')
md_info.shape
```
```
md_info.info()

#set index to "CaseOrder"
md_info.set_index('CaseOrder')
```
```
#check for duplicates
print(md_info.duplicated().value_counts())

md_info.isnull().sum()
```
```
#checks visualization for null values in all the dataset
md_info.isnull().sum()
msno.matrix(md_info, fontsize = 15, labels=True)
plt.title("null records")
plt.show()
```
![null_vizualisation](/assets/SCR-20231123-udza.png)
```
#checks for median, mode, mean values before modifying data in "Overweight"
print("median",md_info['Overweight'].median())
print("mode", md_info['Overweight'].mode())
print("mean",md_info['Overweight'].mean())
#checks for null values before modifying data
plt.hist(md_info['Overweight'])
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()
print("null:",md_info['Overweight'].isnull().sum())
#extract sum of each category group values before modifying the data
sum_by_value = md_info.groupby('Overweight')['Overweight'].count()
print("Sum",sum_by_value)

```
![median vizualisation](/assets/SCR-20231123-ufqi.png)

```
#remove null values in "Overweight"
md_info.dropna(subset=["Overweight"], inplace=True)
#double checks for null values
print("null:",md_info['Overweight'].isnull().sum())
#new histogram with modified data
plt.hist(md_info['Overweight'])
plt.title('is the patient overweight')
plt.ylabel('# of patients')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()
#check original values are not modified after removing nulls
sum_by_value = md_info.groupby('Overweight')['Overweight'].count()
print(sum_by_value)

```
![overweight 2nd vizualisation](/assets/SCR-20231123-uhav.png)

```
#generated the z score and printed zscore on overweight
md_info['Overweight_zscore']=stats.zscore(md_info['Overweight'])
print(md_info[['Overweight','Overweight_zscore']].head)
#generated histogram to detect outliers
plt.hist(md_info['Overweight_zscore'])
plt.xlabel('Overweight zscore')
plt.ylabel('Frequency')
plt.title('Overweight_zscore')
plt.show()

```
![overweight z score](/assets/SCR-20231123-ukjb.png)

```
#imputation of random values from Age min and max
md_info['Age'] = md_info['Age'].apply(lambda x: np.random.uniform(md_info['Age'].min(), md_info['Age'].max()) if pd.isnull(x) else x)

#histogram for "Age" after imputation
age_values = md_info['Age']
plt.hist(age_values)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

#check for null values and median values of the data after imputation
print("Null:",md_info['Age'].isnull().sum())
print("Median:",md_info['Age'].median())
print("Mode", md_info['Age'].mode())
print("Mean",md_info['Age'].mean())
```
![age after imputation](/assets/SCR-20231123-ultb.png)

```
#generated the z score and printed zscore
md_info['age_zscore']=stats.zscore(md_info['Age'])
print(md_info[['Age','age_zscore']].head)
#generated histogram to detect outliers
plt.hist(md_info['age_zscore'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age zscore')
plt.show()
#created Age boxplot as well to check for outliers
boxplot=sns.boxplot(x='Age',data=md_info)
```

![z score age](/assets/SCR-20231123-umpv.png)

```
#histogram for "Children" before imputation
plt.hist(md_info['Children'])
plt.xlabel('# of Children')
plt.ylabel('Frequency')
plt.title('# of Children in the household')
plt.show()
#check for null values and median values of Children before imputation
print("Null:",md_info['Children'].isnull().sum())
print("Median:",md_info['Children'].median())
print("Mode", md_info['Children'].mode())
print("Mean",md_info['Children'].mean())
```
![children before imputation](/assets/SCR-20231123-uocw.png)

```
#since data is posivitely skewed will do an univariate imputation with the median: 
md_info['Children'].fillna(md_info['Children'].median(),inplace=True)

#histogram for "Children" after imputation
md_info['Children']
plt.hist(md_info['Children'])
plt.xlabel('# of Children')
plt.ylabel('Frequency')
plt.title('# of Children in the household')
plt.show()
#check for null values and median values of Children before imputation
print("Null:",md_info['Children'].isnull().sum())
print("Median:",md_info['Children'].median())
print("Mode", md_info['Children'].mode())
print("Mean",md_info['Children'].mean())
```
![Children after imputation](/assets/SCR-20231123-uoxx.png)

```
#generated the z score and printed zscore for Children
md_info['Children_zscore']=stats.zscore(md_info['Children'])
print(md_info[['Children','Children_zscore']].head)
#generated histogram to detect outliers in childrens
plt.hist(md_info['Children_zscore'])
plt.xlabel('z score')
plt.ylabel('Frequency')
plt.title('Children_zscore')
plt.show()
#created Age boxplot as well to check for outliers in childrens
boxplot=sns.boxplot(x='Children',data=md_info)
```

![z score children](/assets/SCR-20231123-upym.png)


do the same for all other variables:






## Data Modeling and Process:

- **Data Preparation:** The data cleaning plan involved checking dataset size, renaming ambiguous column names, identifying and addressing missing values, and outlier detection.
  
- **PCA Application:** I applied Principal Component Analysis to the dataset, focusing on variables from patient surveys, resulting in the identification of two significant principal components.
  
## Deliverables and Significance:

- The project resulted in a cleaned dataset, ready for in-depth analysis, with PCA revealing key components influencing patient readmissions.
- Annotated code provided in “clean_medical_data.ipynb”.
- Cleaned dataset available as “D206cleaned_medical_dataset.csv”.
  
## Limitations and Impact:

Limitations included dataset size and self-reported data, potentially affecting the analysis's accuracy.
These limitations could impact the certainty of findings related to the research question.

This project highlights my skills in data cleaning, PCA application, and analysis within a healthcare context, demonstrating my ability to draw insights from complex datasets.
