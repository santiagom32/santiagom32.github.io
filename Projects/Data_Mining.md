---
layout: default
title: "Data Mining KNN "
---

# Health Care K-nearest Neighbor

## Objective:

To apply the K-nearest neighbor (KNN) classification method to a medical dataset, aiming to predict patient readmission rates based on various health-related variables.

## Research Question Analysis:

The project centered around determining if patient readmission can be predicted using other variables in the dataset. This analysis is crucial for healthcare organizations looking to reduce readmission rates.

## Method Justification:

KNN was chosen for its ability to classify binary outcomes effectively, making it suitable for predicting readmission ('Yes' or 'No'). The method leverages the proximity of data points to predict outcomes, aligning with the categorical nature of the target variable.

## Data Preparation:

The data preparation process involved thorough cleaning, including handling duplicates, null values, and outliers. A range of both continuous (like 'Age' and 'VitD_levels') and categorical variables (like 'HighBlood' and 'Stroke') were selected for the analysis, with one-hot encoding applied to categorical variables for effective processing.

```
#assigned x and y features
x=md_info.drop('ReAdmis',axis=1)
y=md_info['ReAdmis']
x.head()
```
```
#splited the data 80% into training and 20% into test dataset & added constant to both x train and x test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#joined training and testing sets for export
traning_set=pd.concat([x_train,y_train],axis=1)
testing_set=pd.concat([x_test,y_test],axis=1)
#exported train and test sets
traning_set.to_csv('/Users/santiago/Desktop/WGU /MS data analytics/D209 - Data Mining I /D209 task 1/tainingset-D209-1.csv', index = False)
testing_set.to_csv('/Users/santiago/Desktop/WGU /MS data analytics/D209 - Data Mining I /D209 task 1/testingset-D209-1.csv', index = False)
#added constant to te x sets for modeling
x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)
```
```
#created list with odd numbers from 1 to 42
param_grid = {'n_neighbors': list(range(1,43,2))}
knn = KNeighborsClassifier()
#used the gridsearch to find best performing value in previous list with a 10 fold cross validation
grid_search = GridSearchCV(knn, param_grid, cv=10)
grid_search.fit(x_train, y_train)
print('best parameters:',grid_search.best_params_)
print('best score:',grid_search.best_score_)
```
<img width="324" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/e1351c73-b4be-4380-b294-ca7d82eeac37">

```
#Created plot with the accuracies with all neighbors from 1 to 42 t
neighbors= np.arange(1,43)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy[i]=knn.score(x_train, y_train)
    test_accuracy[i]=knn.score(x_test,y_test)
```
-  Accuracy

```
plt.plot(neighbors,test_accuracy,label='testing accuracy')
plt.plot(neighbors,train_accuracy,label='train accuracy')
plt.legend()
plt.xlabel('number of neighbors')
plt.ylabel('accuracy')
plt.show()
```
```
#created model with 37 as best performing n_neighbor value
knn_best=KNeighborsClassifier(n_neighbors=15)
knn_best.fit(x_train,y_train)
y_pred=knn_best.predict(x_test)
knn_best.score(x_test,y_test)
```
<img width="590" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/1cc42075-0e4c-44fd-b06b-a349c5b86a8d">
<img width="590" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/732e4f9b-e220-4b04-be4a-1750bd3d3550">

```
#roc curve
y_pred_prob=knn_best.predict_proba(x_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='K-nn')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
```
<img width="582" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/424677b2-b074-42f2-bfd1-a86718fd44cf">
```
#extracted AUC value
auc = roc_auc_score(y_test, y_pred_prob)
print('AUC',round(auc,4))
```
<img width="99" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/fce944bf-b1ea-4210-8419-4fe09335f9a3">

```
plt.figure(figsize=(8,6))
sns.heatmap(ct_cm_knn, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])

plt.ylabel('actual')
plt.xlabel('Predicted')
plt.title('knn Confusion Matrix ')
plt.show()
```
<img width="649" alt="image" src="https://github.com/santiagom32/santiagom32.github.io/assets/138883598/0049303c-8f3d-4e71-b951-7e16c2ad3ed9">

## Analysis Technique:

The KNN model was trained using a split dataset approach, optimizing for the best n_neighbors value using GridSearchCV. The model's performance was evaluated using accuracy scores, ROC curve analysis, and confusion matrices.

## Findings and Implications:

The model achieved a high accuracy score and AUC, indicating its effectiveness in predicting readmissions. However, a limitation was noted in its inability to quantify the influence of individual variables on the prediction. The results suggest that while the model is reliable in classifying readmissions, further refinement and analysis could enhance its predictive capabilities.

## Recommendations:

Future actions include testing the model with new data to validate its predictive power and refining data collection processes for medical variables to increase analysis accuracy. Additionally, conducting feature selection on explanatory variables could provide deeper insights into their relationships with readmissions.
