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
fpr, tpr, thresholds = roc_curve(y_val,log.predict_proba(X_val)[:,1])         #log_val_pred_probability[:,1])
roc_auc = auc(fpr, tpr)



fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_val_pred_probability[:,1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




