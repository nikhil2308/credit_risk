# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:25:18 2018

@author: Rutuja
"""

#%%Reading the dataframe
import pandas as pd
import numpy as np
#%%
pd.set_option('display.max_columns',None)#view all columns
pd.set_option('display.max_rows',None)#view all rows

proj_data= pd.read_csv(r'C:\Users\hp\Downloads\XYZCorp_LendingData.txt', delimiter='\t')

#%%
#deleting columns having missing values more than 50% 
half_count=len(proj_data)/2
proj_data=proj_data.dropna(thresh=half_count,axis=1)
print(proj_data.isnull().sum())

#%%handling emp_length

proj_data['emp_length'].replace('< 1 year','0 years',inplace=True)
proj_data['emp_length'].replace('10+ years','10 years',inplace=True)
proj_data['emp_length']=proj_data['emp_length'].str.extract('(\d+)').astype(float)
proj_data['emp_length'].value_counts()

#%%
proj_data['emp_length'].fillna(0,inplace=True)
print(proj_data.isnull().sum())

#%%backfilling missing values
colname=['tot_cur_bal','total_rev_hi_lim','tot_coll_amt','collections_12_mths_ex_med','revol_util']
for x in colname:
    proj_data[x].fillna(int(proj_data[x].mean()),inplace=True)
print(proj_data.isnull().sum())

#%%handling date values with mode

colname=['last_pymnt_d','last_credit_pull_d','next_pymnt_d']
for x in colname:
    proj_data[x].fillna((proj_data[x].mode()[0]),inplace=True)
print(proj_data.isnull().sum())

#%%Convert date into int
import datetime
columns=['issue_d','last_pymnt_d','next_pymnt_d','last_credit_pull_d','earliest_cr_line']

for x in columns:
    #proj_data[x]=proj_data[x].map(dt.datetime.toordinal)
    proj_data[x] = pd.to_datetime(proj_data[x], format='%b-%Y')
    proj_data[x]=proj_data[x].astype(str)
    proj_data[x]=proj_data[x].str.replace('\D', '').astype(int)
proj_data[columns].dtypes

#%%label encoding
proj_data.select_dtypes(include=['object']).dtypes
colname=['term','grade','home_ownership','verification_status','pymnt_plan','purpose','initial_list_status','application_type']
#colname=['emp_length']
from sklearn import preprocessing
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    proj_data[x]=le[x].fit_transform(proj_data.__getattr__(x))

#%%
proj_data.drop(['title'], axis = 1, inplace = True)
proj_data.drop(['emp_title'], axis = 1, inplace = True)
proj_data.drop(['sub_grade'], axis = 1, inplace = True)
proj_data.drop(['id'], axis = 1, inplace = True)
proj_data.drop(['member_id'], axis = 1, inplace = True)
proj_data.drop(['addr_state'], axis = 1, inplace = True)
proj_data.drop(['zip_code'], axis = 1, inplace = True)

#%%**********************************************MODEL 1*********************************************************
#train test split

train_data=proj_data[proj_data['issue_d']<=20150501]
test_data=proj_data[proj_data['issue_d']>20150501]

#%%create X and Y variable
X_train=train_data.values[:,:-1]
Y_train=train_data.values[:,-1]
X_test=test_data.values[:,:-1]
Y_test=test_data.values[:,-1]


#%%STANDARDIZE
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
Y_train.astype(int)
Y_test.astype(int)


#%%Building the model
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#%%evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)



#%%*********************************************MODEL 2************************************************
#heatmap
import seaborn as sns
import matplotlib.pyplot as plt

ind_df=proj_data.iloc[:,:-1]
corr_df=ind_df.corr(method="pearson")
a4_dimens=(30,30)
#a4_dimens=(15.7,10.24)
fig , ax=plt.subplots(figsize=a4_dimens)
sns.heatmap(corr_df,cmap="Blues",vmax=0.8,center=0,
            square=True,linewidths=.3,cbar_kws={'shrink':.5},
            annot=True,
            xticklabels=corr_df.columns.values,
            yticklabels=corr_df.columns.values)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.show()

#%%
sns.countplot(x='verification_status',data=proj_data, palette='hls')
plt.show()

proj_data['application_type'].value_counts()
#%%Subsetting columns

proj_20=proj_data.drop(['funded_amnt','funded_amnt_inv','installment','total_pymnt','total_pymnt_inv','out_prncp_inv','collection_recovery_fee','int_rate','total_rev_hi_lim'],axis=1)

proj_20.head()

#%%traintest split
train_data=proj_20[proj_20['issue_d']<=20150501]
test_data=proj_20[proj_20['issue_d']>20150501]

#%%create X and Y variable
X_train=train_data.values[:,:-1]
Y_train=train_data.values[:,-1]
X_test=test_data.values[:,:-1]
Y_test=test_data.values[:,-1]
#%%STANDARDIZE
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
Y_train.astype(int)
Y_test.astype(int)

#%%Building the model
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#%%evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)

Y_pred_class=[]
for value in Y_pred_prob[:,0]:
    if value < 0.70:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)

        
#convert y test to list so that both the inputs have same data structure
#convert y test to list so that both the inputs have same data structure
cfm=confusion_matrix(Y_test.tolist(),Y_pred_class)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test.tolist(),Y_pred_class))

acc=accuracy_score(Y_test.tolist(),Y_pred_class)
print("Accuracy of the model: ",acc)
"""
import numpy as np
for a in np.arange(0,1,0.05):
    predict_mine = np.where(Y_pred_prob[:,0] < a,1,0)
    cfm=confusion_matrix(Y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Error at threshold",a, ":" ,total_err,\
          "type 2 error:",cfm[1,0], "type 1 error", cfm[0,1])
"""
#%% cross validation Logistic regression
classifier=(LogisticRegression())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#%%plot auc and roc(reciever operator characterstics)
from sklearn import metrics
#preds=classifier.pred_proba(X_test)[:,0]
fpr,tpr,threshold=metrics.roc_curve(Y_test.tolist(),Y_pred_class)
auc=metrics.auc(fpr,tpr)
print(auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='lower right')   
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
