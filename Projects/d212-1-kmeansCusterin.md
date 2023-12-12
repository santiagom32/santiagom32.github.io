---
layout: default
title: "d212 kmeansCustering"
---
# d212 kmeansCustering

## Overview
This project explores a medical dataset containing information on 10,000 patients to uncover factors indicating readmission risks using clustering techniques. It aims to classify patients based on their income, initial hospitalization days, total cost, and additional charges.

## Objective
To identify patterns within the selected features (income, initial hospitalization days, total cost, and additional charges) to classify patients into distinct groups and evaluate the accuracy and differences of these classifications.

## Research Question
Can patients be effectively classified based on their income, initial hospitalization days, total cost, and additional charges?

## Technique Justification
Choice of Clustering Technique: K-means, using only continuous variables.
Rationale: K-means is efficient for large datasets and works well with numerical data. It requires less processing power and is suitable when the number of clusters is known or can be estimated.
Assumption: Assumes independence among features and equal contribution to the distance between data points.

## Tools and Libraries:
Pandas for data analysis and manipulation.
Seaborn and matplotlib.pyplot for data visualization.
Plotnine for generating density plots.
StandardScaler and KMeans from sklearn for data scaling and clustering.
silhouette_score from sklearn.metrics to evaluate model accuracy.


## Data Preparation
**Preprocessing Goal:**
The data must contain only continuous variables and scaled values for effective K-means clustering.

<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/56908426-57b8-4f41-b15b-12d70425a6aa">

**Data Variables:**
Income (Continuous)
Initial_days (Continuous)
Total_charge (Continuous)
Additional_charge (Continuous)

<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/936a8a37-9bcc-4a43-be77-7b7729fcd2b2">

**Preparation Steps:**
Import libraries and dataset.
Data exploration and dropping irrelevant features.
Null and duplicate value checks.
<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/ba52bee5-3dd6-49d4-b28c-c1814a3680e6">

Data scaling using StandardScaler.

<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/f7cb55f4-eb2a-4310-8d8b-0a4d70f0270e">


## Analysis
Determining Optimal Clusters:
Method: Analyzed using inertia and silhouette score.
Outcome: Optimal number of clusters identified as two.

<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/140d2946-68c2-4fea-a0a8-125572ec0682">
<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/e8925014-faf3-45b4-9d56-d37c6f923275">


Clustering Analysis Code: Provided separately.

## Data Summary and Implications
Cluster Quality: Silhouette score of 0.4426 indicates reasonably spaced clusters, but not very well-defined.
<img width="350" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/c94f0a0c-40ed-4585-8b3c-44bd8ef7fc3d">

## Analysis Results:
Higher standard deviation in Income and Additional_charge suggests they are less effective for clustering.
Initial_days and Total_charge are well-defined within each cluster.
<img width="456" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/7bd57622-a653-49c0-80e7-18c3edae937a">

Limitation: K-means' assumption of no correlation between features can lead to poorly-defined clusters.

## Recommendations

Based on the analysis, it's recommended to focus on well-defined features like Initial_days and Total_charge for improved clustering. Poorly defined features should be reconsidered or removed to enhance the analysis.
