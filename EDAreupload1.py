#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os


# In[2]:


import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings     # swatting warngs
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1)
rcParams['figure.figsize'] = 12, 6
RANDOM_SEED = 28

LABELS = ["Normal", "Fraud"]


# In[3]:


# train csv

Train=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Train-1542865627584.csv")
Train_Bendata=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Train_Beneficiarydata.csv")
Train_Inpt=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Train_Inpatientdata.csv")
Train_Outpt=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Train_Outpatientdata.csv")

# test csv

Test=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Test-1542969243754.csv")
Test_Bendata=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Test_Beneficiarydata.csv")
Test_Inpt=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Test_Inpatientdata.csv")
Test_Outpt=pd.read_csv(r"C:\Users\dmg27\Desktop\capstoneNYCDSA\Test_Outpatientdata.csv")


# In[4]:


## what does csv look like 

print('Train data :',Train.shape)
print('TrainBen data :',Train_Bendata.shape)
print('TrainInpt :',Train_Inpt.shape)
print('TrainOutpt :',Train_Outpt.shape)

print('Test data :',Test.shape)
print('TestBen data :',Test_Bendata.shape)
print('TestInpt :',Test_Inpt.shape)
print('TestOutpt :',Test_Outpt.shape)


# In[7]:


#check n.a
Train_Bendata.isna().sum()
Test_Bendata.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




