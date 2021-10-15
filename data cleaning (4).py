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


# In[19]:


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


# In[22]:


# adding provider codes cont. 
Train_fullpatprov=pd.merge(Train,Train_fullpat,on='Provider')

Test_fullpatprov=pd.merge(Test,Test_fullpat,on='Provider')


# In[23]:


Test_fullpatprov.head(2)


# In[34]:


#appending train on test

Test_fullpatprov2=Test_fullpatprov

col_merge=Test_fullpatprov.columns

Test_fullpatprov=pd.concat([Test_fullpatprov,
                                               Train_fullpatprov[col_merge]])


# In[37]:


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


# In[38]:


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


# In[39]:


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


# In[40]:


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


# In[41]:


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


# In[42]:


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


# In[43]:


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


# In[44]:


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


# In[45]:


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


# In[46]:


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


# In[47]:


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


# In[48]:


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


# In[49]:


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


# In[50]:


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


# In[51]:


#change n/a to 0 to improve model time

cols1 = Train_fullpatprov.select_dtypes([np.number]).columns
cols2 = Train_fullpatprov.select_dtypes(exclude = [np.number]).columns

Train_fullpatprov[cols1] = Train_fullpatprov[cols1].fillna(value=0)
Test_fullpatprov[cols1]=Test_fullpatprov[cols1].fillna(value=0)


# In[52]:


# trim fat from csv 
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


# In[53]:


# changing data to categor. 

Train_catnull.Gender=Train_catnull.Gender.astype('category')
Test_catnull.Gender=Test_catnull.Gender.astype('category')

Train_catnull.Race=Train_catnull.Race.astype('category')
Test_catnull.Race=Test_catnull.Race.astype('category')


# In[54]:


# adding dummified columns for categor. 

Train_catnull=pd.get_dummies(Train_catnull,columns=['Gender','Race'],drop_first=True)
Test_catnull=pd.get_dummies(Test_catnull,columns=['Gender','Race'],drop_first=True)


# In[56]:


#potential fraud from y/n to 1/0

Train_catnull.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
Train_catnull.head()
Train_catnull.PotentialFraud=Train_catnull.PotentialFraud.astype('int64')
Train_catnull.PotentialFraud.dtypes
Train_catnull.PotentialFraud.min()


# In[57]:


Test_fullpatprov.head(2)


# In[58]:


Train_catnull


# In[59]:


Test_catnull=Test_catnull.iloc[:135392]


# In[60]:


Test_catnull.tail()


# In[61]:


# claims agg by prov

Train_PFcatnull=Train_catnull.groupby(['Provider','PotentialFraud'],as_index=False).agg('sum')
Test_PFcatnull=Test_catnull.groupby(['Provider'],as_index=False).agg('sum')


# In[62]:


#adding target fill column "y"

X=Train_PFcatnull.drop(axis=1,columns=['Provider','PotentialFraud'])
y=Train_PFcatnull['PotentialFraud']


# In[63]:


#scaling to unit variance #nomeans #applying to invis data

sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)

X_teststd=sc.transform(Test_PFcatnull.iloc[:,1:])


# In[64]:


print('X Shape:',X_std.shape)


# In[66]:


#strat y's 

X_train,X_val,y_train,y_val = train_test_split(X_std,y,test_size=0.3,random_state=101,stratify=y,shuffle=True)

print('X_train :',X_train.shape)
print('y_train :',y_train.shape)

print('X_val :',X_val.shape)
print('y_val :',y_val.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




