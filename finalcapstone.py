#!/usr/bin/env python
# coding: utf-8

# DATA IMPORT

# In[ ]:


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


# In[5]:


#check n.a
Train_Bendata.isna().sum()
Test_Bendata.isna().sum()


# In[6]:


#n.a search
Train_Inpt.isna().sum()

Test_Inpt.isna().sum()


# In[7]:


#n.a search


Train_Outpt.isna().sum()

Test_Outpt.isna().sum()


# In[ ]:





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


# In[9]:


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


# In[10]:


#newvar Deceased y/n : 1/0

Train_Bendata.loc[Train_Bendata.DOD.isna(),'Deceased']=0
Train_Bendata.loc[Train_Bendata.DOD.notna(),'Deceased']=1
Train_Bendata.loc[:,'Deceased'].head(7)


Test_Bendata.loc[Test_Bendata.DOD.isna(),'Deceased']=0
Test_Bendata.loc[Test_Bendata.DOD.notna(),'Deceased']=1
Test_Bendata.loc[:,'Deceased'].head(3)


# In[11]:


#putting inpt and outpt together with bendata  and provider data bc easier to deal with

Train_fullpat=pd.merge(Train_Outpt,Train_Inpt,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode']
                              ,how='outer')


Test_fullpat=pd.merge(Test_Outpt,Test_Inpt,
                              left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],
                              right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode']
                              ,how='outer')


Train_fullpat=pd.merge(Train_fullpat,Train_Bendata,left_on='BeneID',right_on='BeneID',how='inner')

Test_fullpat=pd.merge(Test_fullpat,Test_Bendata,left_on='BeneID',right_on='BeneID',how='inner')


# In[12]:


# adding provider codes cont. 
Train_fullpatprov=pd.merge(Train,Train_fullpat,on='Provider')

Test_fullpatprov=pd.merge(Test,Test_fullpat,on='Provider')


# In[13]:


Train_fullpatprov.head()


# In[14]:


#fraud figures based on trans. 

sns.set_style('white',rc={'figure.figsize':(12,8)})
count_classes = pd.value_counts(Train_fullpatprov['PotentialFraud'], sort = True)
print("Percent Distribution of Potential Fraud class:- \n",count_classes*100/len(Train_fullpatprov))
LABELS = ["Not Fraudulant", "Fraudulant"]
#Drawing a barplot
count_classes.plot(kind = 'bar', rot=0,figsize=(10,6))

#Giving titles and labels to the plot
plt.title("Fraud Claim Vulnerability Data Visualized")
plt.xticks(range(2), LABELS)
plt.xlabel("")
plt.ylabel("Possible Fraud Claims ")

plt.savefig('Pt')


# In[15]:


## fraud based on procedure

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax=sns.countplot(x='ClmProcedureCode_1',hue='PotentialFraud',data=Train_fullpatprov
              ,order=Train_fullpatprov.ClmProcedureCode_1.value_counts().iloc[:10].index)

plt.title('Fraud Vulnerable Procedures By Code')
    
plt.show()

plt.savefig('fraudbyproceedure')


# In[266]:


## +10 Claim Diagnosis  invloved in Healthcare Fraud

sns.set(rc={'figure.figsize':(12,8)},style='white')

sns.countplot(x='ClmDiagnosisCode_1',hue='PotentialFraud',data=Train_fullpatprov
              ,order=Train_fullpatprov.ClmDiagnosisCode_1.value_counts().iloc[:10].index)

plt.title('Top 10 Diagnosis Vulnerable to Healthcare Fraud By Code')
plt.show()


# In[16]:


#physician associated fraud 

sns.set(rc={'figure.figsize':(12,8)},style='white')

ax= sns.countplot(x='AttendingPhysician',hue='PotentialFraud',data=Train_fullpatprov
              ,order=Train_fullpatprov.AttendingPhysician.value_counts().iloc[:20].index)

    
plt.title('Physicians Associated with HCF by Code')
plt.xticks(rotation=90)
plt.show()

plt.savefig('PhysicianHCF')


# In[17]:


Train_fullpatprov3=Train_fullpatprov
Train_fullpatprov3.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
Train_fullpatprov3.head()
Train_fullpatprov3.PotentialFraud=Train_fullpatprov.PotentialFraud.astype('int64')
Train_fullpatprov3.PotentialFraud.dtypes
Train_fullpatprov3.PotentialFraud.min()


# In[62]:


Test_fullpatprov.head()


# In[18]:


data123 = Train_fullpatprov3.groupby('Provider').agg({'ClaimID':'count',"PotentialFraud":"mean"})
data123['PotentialFraud'] = data123.PotentialFraud.replace({0:"None",1:"Suspect"})

sns.displot(data = data123, x="ClaimID", log_scale=True,kind='kde',hue='PotentialFraud',legend=False)

xticks= [1,10,100,1000,10000]
plt.xticks(ticks = xticks,labels=xticks)
plt.xlabel("# Claims per Provider")
plt.ylabel('Proportion of Providers')
plt.title("Claims per Provider by Fraud Class")

plt.legend(title="Potential Fraud", labels = ["Yes","No"])
plt.tight_layout()


# #NETWORKX

# In[19]:


pip install networkx


# In[20]:



import pandas as pd
import numpy as np
import networkx as nx


# In[21]:



pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[22]:


data = Train_fullpatprov


# In[23]:


#adds connections columns per provider. #WARNING: USE head() when checking, WILL BREAK PROGRAM/operating system #learned that the hard way

def network_connections(data):
 
    Full_soc_dataProv = data.groupby(['Provider','AttendingPhysician'])['IPAnnualReimbursementAmt'].count().reset_index()
    Full_soc_dataPatient = data.groupby(['Provider','BeneID'])['IPAnnualReimbursementAmt'].count().reset_index()
    
    G1 = nx.Graph()
    G1 = nx.from_pandas_edgelist(Full_soc_dataProv, 'Provider','AttendingPhysician')
    prov_phys_degree = pd.DataFrame(G1.degree)
    prov_phys_degree.columns = ['Provider','AttPhys_Connections']
    prov_phys_degree2 = prov_phys_degree[prov_phys_degree['Provider'].str.contains('PRV')]
    data = pd.merge(prov_phys_degree2, data, how="outer", on="Provider")
    
    G2 = nx.Graph()
    G2 = nx.from_pandas_edgelist(Full_soc_dataPatient, 'Provider','BeneID')
    prov_patient_degree = pd.DataFrame(G2.degree)
    prov_patient_degree.columns = ['Provider','Patient_Connections']
    prov_patient_degree2 = prov_patient_degree[prov_patient_degree['Provider'].str.contains('PRV')]
    data = pd.merge(prov_patient_degree2, data, how="outer", on="Provider")
    
    return data


# In[24]:


data = network_connections(data)


# In[25]:


data.head()


# In[26]:


data['Connec_ratio'] = data['AttPhys_Connections']/data['Patient_Connections']

data.head()


# In[27]:




xval=[-0.5,0,0.5,1]
values = ['-0.5','0','0.5','1']

data.Connec_ratio.plot.density(color='green')
plt.xlabel("Ratio of Attending Physician/Patient Connections")
plt.title('Density Plot of Ratio of Attending Physician/Patient Connections')
plt.xticks(xval,values)
plt.xlim([-0.5, 1])
plt.show()


# In[28]:


data_wide = data.pivot(columns = 'PotentialFraud',
                     values = 'Connec_ratio')
  
# plotting multiple density plot
data_wide.plot.kde(figsize = (8, 6),
                   linewidth = 4)
plt.xlabel("Ratio of Attending Physician/Patient Connections")
plt.ylabel('Density')
plt.title("Connection Proporiton by Fraud Class")
plt.legend(title="Potential Fraud", labels = ["Yes","No"])


# In[30]:


data345 = data.groupby('Provider').agg({'Connec_ratio':'mean',"PotentialFraud":"mean"})

sns.displot(data = data, x="Connec_ratio", log_scale=True,kind='kde',hue='PotentialFraud',legend=False)

#xticks= [0,0.001,0.01,0.1,0.2,0.4,1,2,3]
#plt.xticks(ticks = xticks,labels=xticks)
plt.xlabel("-")
plt.ylabel('-')
plt.title("Connec proporiton by Fraud Class")
plt.xlim([0, 5])

plt.legend(title="Potential Fraud", labels = ["Yes","No"])
plt.tight_layout()


# In[ ]:





# In[31]:


data345.Connec_ratio.plot.density(color='green')
plt.title('Density plot for ratio')
plt.show()


# In[32]:


data345.head()


# In[33]:


data[data['PotentialFraud']==0].nsmallest(1, 'Connec_ratio')


# In[34]:


data[data['PotentialFraud']==0].nlargest(1, 'Connec_ratio')


# In[35]:


import statistics


# In[36]:


data["Patient_Connections"].mean()


# In[37]:


data["AttPhys_Connections"].mean()


# In[38]:


data["InscClaimAmtReimbursed"].sum()


# In[39]:


data.groupby("PotentialFraud").mean()


# In[40]:


data.groupby("PotentialFraud").mean()


# In[41]:


data['Patient_Connections'].std()


# In[ ]:





# In[42]:


#appending train on test

Test_fullpatprov2=Test_fullpatprov

col_merge=Test_fullpatprov.columns

Test_fullpatprov=pd.concat([Test_fullpatprov,
                                               Train_fullpatprov[col_merge]])


# In[43]:


#mean feat by grouping variables and  by prov code

Train_fullpatprov["PerProviderAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerProviderAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerProviderAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerProviderAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerProviderAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerProviderAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerProviderAvg_Age"]=Train_fullpatprov.groupby('Provider')['Age'].transform('mean')
Train_fullpatprov["PerProviderAvg_NoOfMonths_PartACov"]=Train_fullpatprov.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
Train_fullpatprov["PerProviderAvg_NoOfMonths_PartBCov"]=Train_fullpatprov.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')


Test_fullpatprov["PerProviderAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerProviderAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerProviderAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerProviderAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerProviderAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerProviderAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerProviderAvg_Age"]=Test_fullpatprov.groupby('Provider')['Age'].transform('mean')
Test_fullpatprov["PerProviderAvg_NoOfMonths_PartACov"]=Test_fullpatprov.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
Test_fullpatprov["PerProviderAvg_NoOfMonths_PartBCov"]=Test_fullpatprov.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')


# In[44]:


## group by PerBeneID describes transac per beneficiary. one ben can fraud mult prvidrs 

Train_fullpatprov["PerBeneIDAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerBeneIDAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerBeneIDAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerBeneIDAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerBeneIDAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerBeneIDAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')



Test_fullpatprov["PerBeneIDAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerBeneIDAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerBeneIDAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerBeneIDAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerBeneIDAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerBeneIDAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')


# In[45]:


### mean feat by OtherPhysician.

Train_fullpatprov["PerOtherPhysicianAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('OtherPhysician')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerOtherPhysicianAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('OtherPhysician')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerOtherPhysicianAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('OtherPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerOtherPhysicianAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('OtherPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerOtherPhysicianAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('OtherPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerOtherPhysicianAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('OtherPhysician')['OPAnnualDeductibleAmt'].transform('mean')

Test_fullpatprov["PerOtherPhysicianAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('OtherPhysician')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerOtherPhysicianAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('OtherPhysician')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerOtherPhysicianAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('OtherPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerOtherPhysicianAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('OtherPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerOtherPhysicianAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('OtherPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerOtherPhysicianAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('OtherPhysician')['OPAnnualDeductibleAmt'].transform('mean')


# In[46]:


##mean feat by OperatingPhysician

Train_fullpatprov["PerOperatingPhysicianAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')

Test_fullpatprov["PerOperatingPhysicianAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerOperatingPhysicianAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')


# In[47]:


# meanfeat by AttendingPhysician   

Train_fullpatprov["PerAttendingPhysicianAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerAttendingPhysicianAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerAttendingPhysicianAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')


# In[48]:


# mean feat by DiagnosisGroupCode  

Train_fullpatprov["PerDiagnosisGroupCodeAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerDiagnosisGroupCodeAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerDiagnosisGroupCodeAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerDiagnosisGroupCodeAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerDiagnosisGroupCodeAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerDiagnosisGroupCodeAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')

Test_fullpatprov["PerDiagnosisGroupCodeAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerDiagnosisGroupCodeAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerDiagnosisGroupCodeAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerDiagnosisGroupCodeAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerDiagnosisGroupCodeAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerDiagnosisGroupCodeAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')


# In[49]:


# meanfeat by ClmAdmitDiagnosisCode 

Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')

Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')


# In[50]:


# meanfeat by ClmProcedureCode_1 

Train_fullpatprov["PerClmProcedureCode_1Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmProcedureCode_1Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_1Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')


# In[51]:


# meanfeat by ClmProcedureCode_2

Train_fullpatprov["PerClmProcedureCode_2Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_2Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_2Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_2Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_2Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_2Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmProcedureCode_2Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_2Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_2Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_2Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_2Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_2Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')


# In[52]:


# meanfeat by ClmProcedureCode_3

Train_fullpatprov["PerClmProcedureCode_3Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_3Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_3Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_3Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_3Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmProcedureCode_3Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmProcedureCode_3Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_3Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_3Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_3Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_3Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmProcedureCode_3Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')


# In[53]:


# meanfeat by ClmDiagnosisCode_1 

Train_fullpatprov["PerClmDiagnosisCode_1Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmDiagnosisCode_1Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')


# In[54]:


# meanfeat by ClmDiagnosisCode_2

Train_fullpatprov["PerClmDiagnosisCode_2Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmDiagnosisCode_2Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')


# In[55]:


# meanfeat by ClmDiagnosisCode_3

Train_fullpatprov["PerClmDiagnosisCode_3Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmDiagnosisCode_3Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')


# In[56]:


# meanfeat by ClmDiagnosisCode_4

Train_fullpatprov["PerClmDiagnosisCode_4Avg_InscClaimAmtReimbursed"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['InscClaimAmtReimbursed'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_4Avg_DeductibleAmtPaid"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_4Avg_IPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_4Avg_IPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_4Avg_OPAnnualReimbursementAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
Train_fullpatprov["PerClmDiagnosisCode_4Avg_OPAnnualDeductibleAmt"]=Train_fullpatprov.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')


Test_fullpatprov["PerClmDiagnosisCode_4Avg_InscClaimAmtReimbursed"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['InscClaimAmtReimbursed'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_4Avg_DeductibleAmtPaid"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_4Avg_IPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_4Avg_IPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_4Avg_OPAnnualReimbursementAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
Test_fullpatprov["PerClmDiagnosisCode_4Avg_OPAnnualDeductibleAmt"]=Test_fullpatprov.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')


# In[58]:


Test_fullpatprov.head()


# In[59]:


#change n/a to 0 to improve model time #if it keyerrors just keep going

cols1 = Train_fullpatprov.select_dtypes([np.number]).columns
cols2 = Train_fullpatprov.select_dtypes(exclude = [np.number]).columns

Train_fullpatprov[cols1] = Train_fullpatprov[cols1].fillna(value=0)
Test_fullpatprov[cols1]=Test_fullpatprov[cols1].fillna(value=0)


# In[64]:


# trim fat from csv removing unnecessary cats
cols=Train_fullpatprov.columns
cols[:58]

remove_these_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode', 'AdmissionDt',
       'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD',
        'State', 'County']

Train_catnull=Train_fullpatprov.drop(axis=1,columns=remove_these_columns)
Test_catnull=Test_fullpatprov.drop(axis=1,columns=remove_these_columns)


# In[65]:


# changing data to categor. 

Train_catnull.Gender=Train_catnull.Gender.astype('category')
Test_catnull.Gender=Test_catnull.Gender.astype('category')

Train_catnull.Race=Train_catnull.Race.astype('category')
Test_catnull.Race=Test_catnull.Race.astype('category')


# In[66]:


#stop and make second dataset for random forest (non-dummified)
Train_catnull2=Train_catnull
Test_catnull2=Test_catnull


# In[67]:


# adding dummified columns for categor. 

Train_catnull=pd.get_dummies(Train_catnull,columns=['Gender','Race'],drop_first=True)
Test_catnull=pd.get_dummies(Test_catnull,columns=['Gender','Race'],drop_first=True)


# In[68]:


#potential fraud from y/n to 1/0

Train_catnull.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
Train_catnull.head()
Train_catnull.PotentialFraud=Train_catnull.PotentialFraud.astype('int64')
Train_catnull.PotentialFraud.dtypes
Train_catnull.PotentialFraud.min()

#non dummy potential fraud from y/n to 1/0

Train_catnull2.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
Train_catnull2.head()
Train_catnull2.PotentialFraud=Train_catnull.PotentialFraud.astype('int64')
Train_catnull2.PotentialFraud.dtypes
Train_catnull2.PotentialFraud.min()



# In[72]:


#just checking progress
Test_catnull.tail()


# In[73]:


# claims agg by prov

Train_PFcatnull=Train_catnull.groupby(['Provider','PotentialFraud'],as_index=False).agg('sum')
Test_PFcatnull=Test_catnull.groupby(['Provider'],as_index=False).agg('sum')

# claims agg by prov non dummy

Train_PFcatnull2=Train_catnull2.groupby(['Provider','PotentialFraud'],as_index=False).agg('sum')
Test_PFcatnull2=Test_catnull2.groupby(['Provider'],as_index=False).agg('sum')


# In[74]:


#adding target fill column "y"

X=Train_PFcatnull.drop(axis=1,columns=['Provider','PotentialFraud'])
y=Train_PFcatnull['PotentialFraud']

#non dummy

Xrf=Train_PFcatnull2.drop(axis=1,columns=['Provider','PotentialFraud'])
yrf=Train_PFcatnull2['PotentialFraud']


# In[77]:


#strat y's 

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=101,stratify=y,shuffle=True)

print('X_train :',X_train.shape)
print('y_train :',y_train.shape)

print('X_val :',X_val.shape)
print('y_val :',y_val.shape)

#strat y's non dummy

Xrf_train,Xrf_val,yrf_train,yrf_val = train_test_split(Xrf,yrf,test_size=0.3,random_state=101,stratify=yrf,shuffle=True)

print('Xrf_train :',Xrf_train.shape)
print('yrf_train :',yrf_train.shape)

print('Xrf_val :',Xrf_val.shape)
print('yrf_val :',yrf_val.shape)


# In[260]:


# count plot on single categorical variable
sns.countplot(x ='PotentialFraud', data = Train_PFcatnull)
 
# Show the plot
plt.show()


# In[79]:


get_ipython().system('pip install imblearn')


# In[80]:


from imblearn.over_sampling import SMOTE
over_sampler = SMOTE(k_neighbors=2)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
Xrf_res, yrf_res = over_sampler.fit_resample(Xrf_train, yrf_train)
#print(f"Training target statistics: {Counter(y_res)}")
#print(f"Testing target statistics: {Counter(y_test)}")
#print(f"Training target statistics: {Counter(yrf_res)}")
#print(f"Testing target statistics: {Counter(yrf_test)}")


# In[259]:


#checking progress
X_res.head()


# In[81]:


from sklearn.linear_model import LogisticRegressionCV

log = LogisticRegressionCV(cv=10,class_weight='balanced',random_state=123)    

# The "balanced" mode uses the y vals to adjust weights inversely proportional to classfreq


log.fit(X_res,y_res)


# In[82]:


## prob pred: 1 and 0 for X_train and X_val

log_train_prob=log.predict_proba(X_res)
log_val_prob=log.predict_proba(X_val)


# In[83]:


# viz of train/val

fig = plt.figure(figsize=(12,8))

sns.distplot(log.predict_proba(X_res)[:,1],color='blue')
sns.distplot(log.predict_proba(X_val)[:,1],color='green')
plt.title('Visualizing Training Data Against Validation Crosscheck ')
plt.xlim([0, 1])

plt.tight_layout()

plt.show()


# In[84]:


#LOGREG test running ~ .84 AUC, not bad. 

from sklearn.metrics import roc_curve, auc,precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_val,log.predict_proba(X_val)[:,1])         #log_PVval_probability[:,1])
roc_auc = auc(fpr, tpr)



fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_PVval_probability[:,1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[85]:


#checking ROC curve

from sklearn.metrics import roc_curve, auc,precision_recall_curve
fpr, tpr, thresholds = roc_curve(y_val,log.predict_proba(X_val)[:,1])         #log_val_pred_probability[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

for label in range(1,10,1):
    plt.text((10-label)/10,(10-label)/10,thresholds[label*15],fontdict={'size': 14})

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[86]:


fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_val_pred_probability[:,1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[87]:


## checking precision and recall 
precision, recall, _ = precision_recall_curve(y_val, log.predict_proba(X_val)[:,1])

plt.plot(precision,recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Vs Recall')


# In[88]:


## density plot Tpr vs fpr distribution.

fig = plt.figure(figsize=(12,8))

sns.distplot(tpr,color='firebrick')

sns.distplot(fpr,color='darkblue')
plt.title('TPR Vs FPR ')
plt.xlim([-.25, 1.2])

plt.text(0.1,4,'Negatives',color='darkblue')
plt.text(0.7,4,'Positives',color='firebrick')
plt.xlabel('Probability')
plt.ylabel('Distribution')
plt.show()


# In[89]:


#prob thresh to  0.60 # balancing out scores 

log_train60=(log.predict_proba(X_res)[:,1]>0.60).astype(bool)
log_val60=(log.predict_proba(X_val)[:,1]>0.60).astype(bool)   # set threshold as 0.60


#Confusion matrix, Accuracy, sensitivity and specificity

from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,auc

cm0 = confusion_matrix(y_res, log_train60,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(y_val, log_val60,labels=[1,0])
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

KappaValue=cohen_kappa_score(y_val, log_val60)
print("Kappa Value :",KappaValue)
AUC=roc_auc_score(y_val, log_val60)

print("AUC         :",AUC)

print("F1-Score Train  : ",f1_score(y_res, log_train60))

print("F1-Score Val  : ",f1_score(y_val, log_val60))


# In[ ]:





# In[95]:


# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":(1.0000e-04, 1.1112e+00, 2.2223e+00, 3.3334e+00, 4.4445e+00,
       5.5556e+00, 6.6667e+00, 7.7778e+00, 8.8889e+00, 1.0000e+01), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10,verbose=3)
logreg_cv.fit(X_res,y_res)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:





# In[98]:


# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

# define the model
model = logreg_cv3
# fit the model
model.fit(X_res, y_res)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.7f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#sort and label via index


# In[61]:


## Lets predict on Test data

log_test60 = (log.predict_proba(X)[:,1]>0.60).astype(bool)
log_testfinal=pd.DataFrame(log_test60)
log_testfinal.head()


# In[99]:


# Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":(1.0000e-04, 1.1112e+00, 2.2223e+00, 3.3334e+00, 4.4445e+00,
       5.5556e+00, 6.6667e+00, 7.7778e+00, 8.8889e+00, 1.0000e+01), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg = LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10,verbose=2)
logreg_cv.fit(X_res, y_res)
importance = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importance = importance.sort_values(by='Importance', ascending=False)

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:





# In[100]:


#checking new score

logreg2=LogisticRegression(C=6.6667,penalty="l2")
logreg2.fit(X_res,y_res)
print("score",logreg2.score(X_res,y_res))


# In[ ]:





# In[107]:


#best version of model

logreg_cv3=logreg_cv.best_estimator_


# #LOGREG IMPORTANCE

# In[106]:


plt.bar(x=importance['Attribute'], height=importance['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.gcf().set_size_inches(20,5)
plt.show()


# In[ ]:





# In[191]:


#1/0 -> y/n
Replacement = {1:'Yes',0:'No'}

Labels=log_testfinal[0].apply(lambda x : Replacement[x])
Labels.value_counts()    #count


# In[192]:


## make file

sub_log=pd.DataFrame({"Provider":Test_PFcatnull.Provider})
sub_log['PotentialFraud']=Labels
sub_log.shape

#writefile

sub_log.to_csv("Submission_Logistic_Regression_F1_60_Threshold_60Prcnt.csv",index=False)


# In[193]:


Test_PFcatnull2.head()


# In[194]:


XTest=Test_PFcatnull.drop(axis=1,columns=['Provider'])


# In[196]:


log_final=(logreg_cv3.predict_proba(X_res)[:,1]>0.60).astype(bool)


# In[190]:


#Confusion matrix, Accuracy, sensitivity and specificity for imprvoved model

from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,auc

cm0 = confusion_matrix(y_res, log_final,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(y_val, log_val60,labels=[1,0])
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

KappaValue=cohen_kappa_score(y_val, log_val60)
print("Kappa Value :",KappaValue)
AUC=roc_auc_score(y_val, log_val60)

print("AUC         :",AUC)

print("F1-Score Train  : ",f1_score(y_res, log_tfinal))

print("F1-Score Val  : ",f1_score(y_val, log_val60))


# In[258]:


# Create a confusion matrix of LOGREG
cnf_matrix = confusion_matrix(y_res, log_final)

# Create heatmap from the confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[False, True] # name  of classes
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted label')
tick_marks = [0.5, 1.5]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# In[197]:


#random forest time! 

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=279,max_depth=5) 
rfc.fit(Xrf_res,yrf_res)


# In[198]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(yrf_val, rfc.predict_proba(Xrf_val)[:,1])
roc_auc = auc(fpr, tpr)

fpr, tpr, thresholds =roc_curve(y_val, log.predict_proba(X_val)[:,1],pos_label=1)     #log_PVval_probability[:,1])
roc_aucRF = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[200]:


# thresh to 0.5

rfc_PVtrain = (rfc.predict_proba(Xrf_train)[:,1]>0.5).astype(bool)   
rfc_PVval = (rfc.predict_proba(Xrf_val)[:,1]>0.5).astype(bool)


#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,roc_curve

cm0 = confusion_matrix(yrf_train, rfc_PVtrain,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(yrf_val, rfc_PVval,labels=[1,0])
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
AUC=roc_auc_score(yrf_val, rfc_PVval)
print("AUC         :",AUC)


print("F1-Score Train",f1_score(y_train,rfc_PVtrain))
print("F1-Score Validation : ",f1_score(yrf_val, rfc_PVval))


# In[218]:


#important features 

feature_list = list(Xrf_train.columns)
# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list[1:], importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
print('Top -20 features impacting Random forest model and their importance score :- \n',)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:15]];


# In[236]:


def plot_feature_importance(importance,names,model_type):

#Create arrays from feature importance and feature names
    feature_importance = np.array(importances)
    feature_names = np.array(names)

#Create a DataFrame using a Dictionary
    impdata={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(impdata)

#Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#Define size of bar plot
    plt.figure(figsize=(10,8))
#Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.gcf().set_size_inches(12,20)
    #plt.savefig('rfcapstoneplot2.png', dpi=300, bbox_inches='tight')
    plt.legend()




plot_feature_importance(best_clf.feature_importances_,Xrf_train.columns,'RANDOM FOREST')


# In[202]:


#testing the real thing

rfc_PVtest = rfc.predict(Xrf)
rfc_PVtest=pd.DataFrame(rfc_PVtest)
rfc_PVtest.head(2)


# In[203]:


#1/0 -> y/n

Replacement = {1:'Yes',0:'No'}

Labels=rfc_PVtest[0].apply(lambda x : Replacement[x])


# In[213]:


# Define Parameters
max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, cv = 5)
grid_results = grid.fit(Xrf_train, yrf_train)

# Summarize the results in a readable format
print("Best: {0}, using {1}".format(grid_results.cv_results_['mean_test_score'], grid_results.best_params_))
results_df = pd.DataFrame(grid_results.cv_results_)
results_df


# In[216]:


# Extract the best decision forest 
best_clf = grid_results.best_estimator_
y_pred = best_clf.predict(Xrf_train)

# Create a confusion matrix
cnf_matrix = confusion_matrix(yrf_train, y_pred)

# Create heatmap from the confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[False, True] # name  of classes
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = [0.5, 1.5]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# In[217]:


best_clf2 = (best_clf.predict_proba(Xrf_train)[:,1]>0.5).astype(bool)   
#Confusion matrix, Accuracy, sensitivity and specificity improved model
from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score,roc_auc_score,f1_score,roc_curve

cm0 = confusion_matrix(yrf_train, best_clf2,labels=[1,0])
print('Confusion Matrix Train : \n', cm0)

cm1 = confusion_matrix(yrf_val, rfc_PVval,labels=[1,0])
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
AUC=roc_auc_score(yrf_val, rfc_PVval)
print("AUC         :",AUC)


print("F1-Score Train",f1_score(y_train,best_clf2))
print("F1-Score Validation : ",f1_score(yrf_val, rfc_PVval))


# In[ ]:


#going to compare with xgboost here and next cell

from collections import Counter
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pprint


# In[ ]:


clfs = {
    #'mnb': MultinomialNB(),
    'lr': LogisticRegression(class_weight='balanced'),
    'xgb': XGBClassifier(booster='gbtree')
}


# In[ ]:


#setting f1 score print 
f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_train, y_train)
    y_pred =((clf.predict_proba(X_val)[:,1]>0.5).astype(bool))
    f1_scores[clf_name] = f1_score(y_pred, y_val)


# In[ ]:





# In[ ]:





# In[ ]:




