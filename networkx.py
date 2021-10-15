#!/usr/bin/env python
# coding: utf-8

# In[130]:


pip install networkx


# In[131]:



import pandas as pd
import numpy as np
import networkx as nx


# In[132]:



pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[135]:


data = Train_fullpatprov


# In[136]:


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


# In[139]:


data = network_connections(data)


# In[140]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:




