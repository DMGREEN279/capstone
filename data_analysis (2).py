#!/usr/bin/env python
# coding: utf-8

# In[67]:


from sklearn.linear_model import LogisticRegressionCV

log = LogisticRegressionCV(cv=10,class_weight='balanced',random_state=123)    

# The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
#in the input data as ``n_samples / (n_classes * np.bincount(y))``.

log.fit(X_train,y_train)


# In[68]:


## prob pred: 1 and 0 for X_train and X_val

log_train_prob=log.predict_proba(X_train)
log_val_prob=log.predict_proba(X_val)


# In[72]:


# viz of train/val

fig = plt.figure(figsize=(12,8))

sns.distplot(log.predict_proba(X_train)[:,1],color='blue')
sns.distplot(log.predict_proba(X_val)[:,1],color='green')
plt.title('Visualizing Training Data Against Validation Crosscheck ')
plt.xlim([0, 1])

plt.tight_layout()

plt.show()


# In[77]:


#LOGREG test running ~ 92-93 AUC, not bad. 

from sklearn.metrics import roc_curve, auc,precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_val,log.predict_proba(X_val)[:,1])         #log_PVval_probability[:,1])
roc_auc = auc(fpr, tpr)



fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_PVval_probability[:,1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[94]:


#prob thresh to  0.65 # balancing out scores 

log_train65=(log.predict_proba(X_train)[:,1]>0.65).astype(bool)
log_val65=(log.predict_proba(X_val)[:,1]>0.65).astype(bool)   # set threshold as 0.65


#Confusion matrix, Accuracy, sensitivity and specificity

from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,auc

cm0 = confusion_matrix(y_train, log_train65,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(y_val, log_val65,labels=[1,0])
print('Confusion Matrix Val: \n', cm1)

total0=sum(sum(cm0))
total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy0=(cm0[0,0]+cm0[1,1])/total0
print ('Accuracy Train: ', accuracy0)

accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy Val: ', accuracy1)

sensitivity0 = cm0[0,0]/(cm0[0,0]+cm0[0,1])
print('Sensitivity Train : ', sensitivity0 )

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity Val: ', sensitivity1 )


specificity0 = cm0[1,1]/(cm0[1,0]+cm0[1,1])
print('Specificity Train: ', specificity0)

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity Val: ', specificity1)

KappaValue=cohen_kappa_score(y_val, log_val65)
print("Kappa Value :",KappaValue)
AUC=roc_auc_score(y_val, log_val65)

print("AUC         :",AUC)

print("F1-Score Train  : ",f1_score(y_train, log_train65))

print("F1-Score Val  : ",f1_score(y_val, log_val65))


# In[83]:


## Lets predict on Test data

log_test65 = (log.predict_proba(X_teststd)[:,1]>0.60).astype(bool)
log_testfinal=pd.DataFrame(log_test65)
log_testfinal.head(2)


# In[84]:


#1/0 -> y/n
Replacement = {1:'Yes',0:'No'}

Labels=log_testfinal[0].apply(lambda x : Replacement[x])
Labels.value_counts()    #count


# In[87]:


## make file

sub_log=pd.DataFrame({"Provider":Test_PFcatnull.Provider})
sub_log['PotentialFraud']=Labels
sub_log.shape

#writefile

sub_log.to_csv("Submission_Logistic_Regression_F1_60_Threshold_60Prcnt.csv",index=False)


# In[88]:


#random forest time! 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=279,max_depth=5) 
rfc.fit(X_train,y_train)


# In[89]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_val, rfc.predict_proba(X_val)[:,1])
roc_auc = auc(fpr, tpr)

fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_PVval_probability[:,1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[95]:


# thresh to 0.5

rfc_PVtrain = (rfc.predict_proba(X_train)[:,1]>0.5).astype(bool)   
rfc_PVval = (rfc.predict_proba(X_val)[:,1]>0.5).astype(bool)


#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,roc_curve

cm0 = confusion_matrix(y_train, rfc_PVtrain,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(y_val, rfc_PVval,labels=[1,0])
print('Confusion Matrix Test: \n', cm1)

total0=sum(sum(cm0))
total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy0=(cm0[0,0]+cm0[1,1])/total0
print ('Accuracy Train : ', accuracy0)

accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy Test : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

KappaValue=cohen_kappa_score(y_val, rfc_PVval)
print("Kappa Value :",KappaValue)
AUC=roc_auc_score(y_val, rfc_PVval)
print("AUC         :",AUC)


print("F1-Score Train",f1_score(y_train,rfc_PVtrain))
print("F1-Score Validation : ",f1_score(y_val, rfc_PVval))


# In[96]:


#important features 

feature_list = list(Test_PFcatnull.columns)
# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list[1:], importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
print('Top -20 features impacting Random forest model and their importance score :- \n',)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:15]];


# In[99]:


#testing the real thing

rfc_PVtest = rfc.predict(X_teststd)
rfc_PVtest=pd.DataFrame(rfc_PVtest)
rfc_PVtest.head(2)


# In[100]:


#1/0 -> y/n

Replacement = {1:'Yes',0:'No'}

Labels=rfc_PVtest[0].apply(lambda x : Replacement[x])


# In[107]:


#going to compare with xgboost here and next cell

from collections import Counter
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pprint


# In[109]:


clfs = {
    #'mnb': MultinomialNB(),
    'lr': LogisticRegression(class_weight='balanced'),
    'xgb': XGBClassifier(booster='gbtree')
}


# In[110]:


#setting f1 score print 
f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred =((clf.predict_proba(X_val)[:,1]>0.5).astype(bool))
    f1_scores[clf_name] = f1_score(y_pred, y_val)


# In[111]:


pprint.pprint(f1_scores)
#LR apparently performing well here 


# In[125]:


Train_fullpatprov.head()


# In[ ]:




