#!/usr/bin/env python
# coding: utf-8

# In[8]:


##Changing to binary for yes/no 1/0

Train_Bendata = Train_Bendata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Train_Bendata = Train_Bendata.replace({'RenalDiseaseIndicator': 'Y'}, 1)

Test_Bendata = Test_Bendata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Test_Bendata = Test_Bendata.replace({'RenalDiseaseIndicator': 'Y'}, 1)


# In[10]:


## age

Train_Bendata['DOB'] = pd.to_datetime(Train_Bendata['DOB'] , format = '%Y-%m-%d')
Train_Bendata['DOD'] = pd.to_datetime(Train_Bendata['DOD'],format = '%Y-%m-%d',errors='ignore')
Train_Bendata['Age'] = round(((Train_Bendata['DOD'] - Train_Bendata['DOB']).dt.days)/365)


Test_Bendata['DOB'] = pd.to_datetime(Test_Bendata['DOB'] , format = '%Y-%m-%d')
Test_Bendata['DOD'] = pd.to_datetime(Test_Bendata['DOD'],format = '%Y-%m-%d',errors='ignore')
Test_Bendata['Age'] = round(((Test_Bendata['DOD'] - Test_Bendata['DOB']).dt.days)/365)

Train_Bendata.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Train_Bendata['DOB']).dt.days)/365),
                                 inplace=True)


Test_Bendata.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Test_Bendata['DOB']).dt.days)/365),
                                 inplace=True)


# In[12]:


#newvar Deceased y/n : 1/0

Train_Bendata.loc[Train_Bendata.DOD.isna(),'Deceased']=0
Train_Bendata.loc[Train_Bendata.DOD.notna(),'Deceased']=1
Train_Bendata.loc[:,'Deceased'].head(7)


Test_Bendata.loc[Test_Bendata.DOD.isna(),'Deceased']=0
Test_Bendata.loc[Test_Bendata.DOD.notna(),'Deceased']=1
Test_Bendata.loc[:,'Deceased'].head(3)


# In[ ]:





# In[ ]:





# In[ ]:




