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
[null_vizualisation](/assets/SCR-20231123-udza.png)
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
[median vizualisation](/assets/SCR-20231123-ufqi.png)

```


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
