#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import f1_score
from scipy.stats import norm
import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import argmax
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from pandas.plotting import scatter_matrix

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn_pandas import DataFrameMapper
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint

df = pd.read_csv("Internal_dataset_final.csv")
df.head()
X = df.drop("results",axis=1)
y = df["results"]
X = X.drop(X.columns[[0,1,2,3,26,27,28]],axis =1)
X

df2 = pd.read_csv("External_dataset_final.csv")
X2 = df2.drop("results",axis=1)
y2 = df2["results"]
X2 = X2.drop(X2.columns[[0,1,2,3,26,27,28]],axis =1)
X2

#logistic regression#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
mean_fpr = np.linspace(0, 1, 100)
lr_param = {'max_iter': [5000], 'class_weight': [None]}
lr_model = LogisticRegression()
model_name = 'Logistic Regression'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
df_results2 = pd.DataFrame(columns=['Model','percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'F1 score', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=15)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=lr_model, param_grid=lr_param, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)

best_lr_model = grid_search.best_estimator_
coefs = best_lr_model.coef_.flatten()
factor_names = X1.columns.values

y_pred = grid_search.predict(X_test1)
y_score1 = grid_search.predict_proba(X_test1)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score1)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score1 >= best_thresh).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score1))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)

print(f'Percentile: {15}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {20})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


df_coef = pd.DataFrame({'factor': factor_names, 'coefficient': coefs})

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_prob_pred)
for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score1)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score

    df_results = df_results.append({'percentile': str(20),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'Logistic regression', 'percentile': str(20),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score1)}, ignore_index=True)

fpr1 = fpr
tpr1 = tpr


# In[76]:


df_results2


# In[77]:


constant = best_lr_model.intercept_
print(f"Constant (Intercept): {constant}")
with open('constant.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Constant'])
    writer.writerow([constant])

print("Constant (Intercept) saved to constant.csv")
df_coef
df_coef = df_coef.sort_values(by='coefficient', ascending= True)
df_coef
plt.barh(df_coef['factor'], df_coef['coefficient'], color='gray') 
plt.xticks(rotation=0)
plt.xlabel('Coefficient')
plt.ylabel('Input variables')
plt.title('Coefficients of logistic regression')
plt.savefig("Coefficient_sample.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
best_thresh


# In[78]:


df_coef.to_csv('Combined-lr_coef.csv')


# In[80]:


#SVM#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
svcparam = {'C': [0.1],  'gamma' : [0.1], 'probability' : [True]}
svc_model = SVC()
model_name = 'SVM'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
#df_results2 = pd.DataFrame(columns=['Model','percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'F1 score', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=100)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=svc_model, param_grid=svcparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score2 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score2 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score2)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score2 >= best_thresh).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score2))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)

f1 = f1_score(y_test, y_prob_pred)

print(f'Percentile: {100}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {100})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


df_coef = pd.DataFrame({'factor': factor_names, 'coefficient': coefs})

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score2)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score
        df_results = df_results.append({'percentile': str(100),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'SVM', 'percentile': str(100),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score2)}, ignore_index=True)

fpr2 = fpr
tpr2 = tpr


# In[82]:


#RF#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
rfparam = {'class_weight': [None], 'criterion': ['gini'], 'max_depth': [7], 'max_features': ['sqrt'], 'min_samples_leaf': [2], 'min_samples_split': [5], 'n_estimators': [1000]}
rf_model = RandomForestClassifier()
model_name = 'RF'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
#df_results2 = pd.DataFrame(columns=['percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=rf_model, param_grid=rfparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)
grid_search

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score3 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score3 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score3)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score3 >= best_thresh).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score3))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)
f1 = f1_score(y_test, y_prob_pred)
print(f'Percentile: {100}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {50})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


df_coef = pd.DataFrame({'factor': factor_names, 'coefficient': coefs})

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score3)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score

    df_results = df_results.append({'percentile': str(15),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'Random forest', 'percentile': str(100),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score3)}, ignore_index=True)

fpr3 = fpr
tpr3 = tpr


# In[86]:


#LGBM#
tprs = []
aucs = []
y_preds = []
y_tests = []
param = []
lgbmparam = {'num_leaves': [7], 'learning_rate': [0.1], 'feature_fraction': [0.5],'bagging_fraction': [0.8], 'bagging_freq': [3]} 
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', verbose = -1,  metric='auc', random_state = 0)
model_name = 'LGBM'

df_results = pd.DataFrame(columns=['df2_value', 'row_num', 'result', 'pos_neg', 'probability'])
#df_results2 = pd.DataFrame(columns=['percentile','True positives','True negatives', 'False positives', 'False negatives', 'Accuracy','Sensitivity', 'Specificity', 'ROC AUC'])

X_train = X
X_test = X2
y_train = y
y_test = y2
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
selected_features = X.columns.values[select.get_support()]
X1 = pd.DataFrame(X_train1, columns=selected_features)


grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=lgbmparam, cv=5, scoring='roc_auc')
grid_search.fit(X_train1, y_train)

y_pred = grid_search.predict(X_test1)
if hasattr(grid_search, 'predict_proba'):
    y_score4 = grid_search.predict_proba(X_test1)[:, 1]
else:
    y_score4 = y_pred
    
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score4)

# get the best threshold
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

y_prob_pred = (y_score4 >= best_thresh).astype(bool)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
aucs.append(roc_auc_score(y_test, y_score4))
y_preds.extend(y_prob_pred)
y_tests.extend(y_test)
f1 = f1_score(y_test, y_prob_pred)
print(f'Percentile: {50}')
print(f'Accuracy Score: {accuracy_score(y_tests, y_preds)}')
print(f'AUC Score: {np.mean(aucs)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_tests, y_preds)}\n')
sns.heatmap(confusion_matrix(y_tests, y_preds), annot=True, cmap='Blues')
plt.title(f'Confusion Matrix for {model_name} (Percentile: {50})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


df_coef = pd.DataFrame({'factor': factor_names, 'coefficient': coefs})

cm = confusion_matrix(y_test, y_prob_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

for idx, (actual, predicted, score) in enumerate(zip(y_test, y_prob_pred, y_score4)):
    if actual == 1 and predicted == 1:
        result = 'Correct'
        pos_neg = 'TP'
        prob = score
    elif actual == 0 and predicted == 0:
        result = 'Correct'
        pos_neg = 'TN'
        prob = score
    elif actual == 0 and predicted == 1:
        result = 'Incorrect'
        pos_neg = 'FP'
        prob = score
    else:
        result = 'Incorrect'
        pos_neg = 'FN'
        prob = score

    df_results = df_results.append({'percentile': str(50),'df2_value': X_test.iloc[idx],'row_num': idx,'result': result,'pos_neg': pos_neg,'probability': prob}, ignore_index=True)

df_results2 = df_results2.append({'Model': 'Light GBM', 'percentile': str(50),'True positives': tp,'True negatives': tn,'False positives': fp,'False negatives': fn,'Accuracy': accuracy_score(y_tests, y_preds),
                            'Sensitivity': sensitivity,'Specificity': specificity, 'F1 score': f1,'ROC AUC': roc_auc_score(y_test, y_score4)}, ignore_index=True)

fpr4 = fpr
tpr4 = tpr


# In[88]:


df_results2.to_csv('2023_results2.csv', index=False)


# In[92]:


np.save('fpr1.npy', fpr1)
np.save('fpr2.npy', fpr2)
np.save('fpr3.npy', fpr3)
np.save('fpr4.npy', fpr4)
np.save('tpr1.npy', tpr1)
np.save('tpr2.npy', tpr2)
np.save('tpr3.npy', tpr3)
np.save('tpr4.npy', tpr4)


# In[93]:


data = {'y_test': y_test, 'y_score1': y_score1, 'y_score2': y_score2,'y_score3': y_score3,'y_score4': y_score4}


# In[94]:


df = pd.DataFrame(data)
file_name = 'y_test_and_y_score.csv'
df.to_csv(file_name, index=True) 


# In[96]:


select = SelectPercentile(percentile=15)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
selected_features = X.columns.values[select.get_support()]
selected_features


# In[97]:


select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
selected_features = X.columns.values[select.get_support()]
selected_features


# In[99]:


select = SelectPercentile(percentile=100)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
selected_features = X.columns.values[select.get_support()]
selected_features


# In[100]:


#ROC曲線を描き、AUCを算出
import seaborn as sns
sns.set_palette("Accent") # カラーパレットを設定
plt.plot(fpr1, tpr1, label='Combined-LR (AUC= {:.3f})'.format(auc(fpr1, tpr1)), color=sns.color_palette()[0])
plt.plot(fpr2,tpr2,label='Combined-SVM (AUC= {:.3f})'.format(auc(fpr2, tpr2)), color=sns.color_palette()[2])
plt.plot(fpr3,tpr3,label='Combined-RF (AUC= {:.3f})'.format(auc(fpr3, tpr3)), color=sns.color_palette()[4])
plt.plot(fpr4,tpr4,label='Combined-LGBM (AUC= {:.3f})'.format(auc(fpr4, tpr4)), color=sns.color_palette()[5])
plt.plot([0,0,1], [0,1,1], linestyle='--', color = 'gray')
plt.plot([0, 1], [0, 1], linestyle='--', color = 'gray')
plt.legend()
plt.xlabel('false positive rate (FPR)')
plt.ylabel('true positive rate (TPR)')
plt.savefig("ROC_final.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
plt.show()


# In[132]:


df_coef
df_coef = df_coef.sort_values(by='coefficient', ascending= True)
df_coef
plt.barh(df_coef['factor'], df_coef['coefficient'], color='cornflowerblue') 
plt.xticks(rotation=0)
plt.xlabel('Coefficient')
plt.ylabel('Input variables')
plt.title('Coefficients of logistic regression')
plt.savefig("Coefficient_sample2.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
best_thresh


# In[ ]:





# In[ ]:




