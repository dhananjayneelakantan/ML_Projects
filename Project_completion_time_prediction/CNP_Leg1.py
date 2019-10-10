#!/usr/bin/env python
# coding: utf-8

# In[6]:


#to support python2 and python3
from __future__ import division, print_function, unicode_literals
#Common imports
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 
#to make this notebooks output stable across runs
np.random.seed(42)

#remove warnings
import warnings 
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#Variabletype Initialisation
stringvar=['_channel','createdBy','_user.login','_transactionContext.currentSection','_validationContext','_workflow.currentState.subType','_workflow.inboxItem.assignedBy','_workflow.isError','assignTo','BatteryCount','caCity','caContactName','caZip','code','code1','consultantName','copyOver','Customer Rate Code','customer.circuit','customer.city','customer.esiid','customer.fuseDetails','customer.meterId','customer.revenueAreaCode','customer.section','customer.streetNumber','customer.zipCode','customerTitle','dba','ecName','entityType','exportPower','externalCode','interconnectionMode','noOfPhase','organization.code','ownershipInfo','parallelType','pmName','propertyOwnerCheck','reportExcessGeneration','serviceCenter','tlmHighVoltage','tlmLowVoltage','tlmPhase','tlmXfrmrType','updatedBy','workorderCode']
floatvar=['invCapacity','inverterEfficiency','inverterQuantity','kvar','solarQuantity','tlmMeterCnt']
datetimevar=['agreementEffectiveDate','applicationReceivedDate','applicationRejectedDate','applicationResubmittedDate','cancelledDate','createdAt','designApprovedDate','inspectionCompletedDate','inspectionRejectedDate','ptoDate','removedDate','serviceDesiredDate','StatusDate']

#Leg Selection
leg='Days_created_to_application'

#Variable initialisation
num=['invCapacity','inverterEfficiency','inverterQuantity','kvar','solarQuantity','tlmMeterCnt']
cat=['_workflow.inboxItem.assignedBy','_user.login','organization.code','createdBy','assignTo','caCity','caZip','consultantName','customer.city','customer.zipCode','dba','ecName','pmName','updatedBy']

def Train(path):
    location = path
    #File read and typecast
    df = pd.read_csv(location)

    for col in ['applicationReceivedDate']:
        df[col] = df[col].astype('datetime64')

    df.sort_values(by='applicationReceivedDate')
    df = df[(df['applicationReceivedDate'] > '2019/01/01')]

    #Changing dtype
    for col in [stringvar]:
        df[col] = df[col].astype('str')

    for col in [floatvar]:
        df[col] = df[col].astype('float64')   

    for col in [datetimevar]:
        df[col] = df[col].astype('datetime64')   

    pto_df_2019 = df

    
    #Dataframe selector
    from sklearn.impute import SimpleImputer
    from sklearn import preprocessing

    from sklearn.base import BaseEstimator, TransformerMixin
    class DataFrameSelector(BaseEstimator,TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self,X):
            return X[self.attribute_names].values
    
    #Attribute adder for data pre_processing - days creation
    createdAt_ix, designApprovedDate_ix, applicationReceivedDate_ix, inspectionCompletedDate_ix, agreementEffectiveDate_ix  , ptoDate_ix , StatusDate_ix= [
        list(pto_df_2019.columns).index(col)  
        for col in ("createdAt","designApprovedDate", "applicationReceivedDate", "inspectionCompletedDate", "agreementEffectiveDate","ptoDate","StatusDate")]

    from sklearn.preprocessing import FunctionTransformer

    def add_extra_features(X, compute_day_features=True):
        if compute_day_features:
            Days_created_to_application=(X['applicationReceivedDate']-X['createdAt']).dt.days
            Days_application_to_design=(X['designApprovedDate']-X['applicationReceivedDate']).dt.days
            Days_design_to_inspection=(X['inspectionCompletedDate']-X['designApprovedDate']).dt.days
    #inspection completed - inspection submitted should be added
            Days_inspection_to_agreement=(X['agreementEffectiveDate']-X['inspectionCompletedDate']).dt.days
            Days_agreement_to_pto=(X['ptoDate']-X['agreementEffectiveDate']).dt.days

            Days_recieved_to_status=(X['StatusDate']-X['applicationReceivedDate']).dt.days

            Days_created_to_design=(X['designApprovedDate']-X['createdAt']).dt.days
            Days_created_to_inspection=(X['inspectionCompletedDate']-X['createdAt']).dt.days
            Days_created_to_agreement=(X['agreementEffectiveDate']-X['createdAt']).dt.days
            Days_created_to_pto=(X['ptoDate']-X['createdAt']).dt.days
            return np.c_[X,Days_created_to_application, Days_application_to_design, Days_design_to_inspection,
                         Days_inspection_to_agreement,Days_agreement_to_pto,Days_recieved_to_status,Days_created_to_design,Days_created_to_inspection,Days_created_to_agreement,Days_created_to_pto]


        else:
            return np.c_[X]

    attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                     kw_args={"compute_day_features": True})


    timeline_extra_attribs = attr_adder.fit_transform(pto_df_2019)
    timeline_extra_attribs = pd.DataFrame(
        timeline_extra_attribs,
        columns=list(pto_df_2019.columns)+["Days_created_to_application","Days_application_to_design","Days_design_to_inspection","Days_inspection_to_agreement","Days_agreement_to_pto","Days_recieved_to_status","Days_created_to_design","Days_created_to_inspection","Days_created_to_agreement","Days_created_to_pto"],
        index=pto_df_2019.index)

    timeline_targetimpute = timeline_extra_attribs.dropna(subset=[leg])



    #Testing and training set creation. 
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(timeline_targetimpute, test_size=0.2, random_state=42)

    #Set label for train and test set 
    timeline = train_set.drop(leg, axis=1)
    timeline_labels = train_set[leg].copy()
    timeline_test = test_set.drop(leg, axis=1)
    timeline_test_labels=test_set[leg].copy()

    #NUMERICAL AND CATEGORICAL VARIABLE INITIALISATION - use initialisation with dtype
    timeline_cat=timeline[cat]
    timeline_num=timeline[num]


    #Number and category pipeline creation
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import OneHotEncoder

    num_attribs = list(timeline_num)
    cat_attribs = list(timeline_cat)


    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer',Imputer(strategy="mean",axis=0)),
        ('std_scaler',StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer',SimpleImputer(strategy="constant",fill_value="Unknown", verbose=0, copy=True, add_indicator=True)),
        ('onehotencoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),

    ])


    #Feature union of different pipelines into full pipeline
    from sklearn.pipeline import FeatureUnion

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),    
    ])


    timeline_prepared = full_pipeline.fit_transform(timeline)


    # #Trying regression and prediction on some unseen test data
    # attributes = num_attribs + cat_attribs
    # some_data = timeline_test.iloc[:5]
    # some_labels = timeline_test_labels.iloc[:5]
    # some_data_prepared = full_pipeline.transform(some_data)


    #MODELS
    #Random Forest (this is used to find important features too!)
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_leaf_nodes=16, n_jobs=-1)
    forest_reg.fit(timeline_prepared, timeline_labels)


    #Linear SVR
    from sklearn.svm import LinearSVR
    svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg1.fit(timeline_prepared, timeline_labels)


    #SGD with lasso
    from sklearn.linear_model import SGDRegressor
    sgd_ridge_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l1", random_state=42)
    sgd_ridge_reg.fit(timeline_prepared,timeline_labels)


    # from sklearn.model_selection import cross_val_score
    # svm_scores = cross_val_score(svm_reg1, timeline_prepared, timeline_labels,scoring="r2", cv=5)
    # sgd_scores = cross_val_score(sgd_ridge_reg, timeline_prepared, timeline_labels,scoring="r2", cv=5)
    # forest_scores = cross_val_score(forest_reg, timeline_prepared, timeline_labels,scoring="r2", cv=5)

    # print("\nforest",forest_scores.mean())
    # print("svm",svm_scores.mean())
    # print("sgd",sgd_scores.mean())


    #Save the model as a pickle in a file 
    svmj=joblib.dump(svm_reg1, 'svm_model_leg1.pkl') 
    forj=joblib.dump(forest_reg, 'forest_model_leg1.pkl') 
    sgdj=joblib.dump(sgd_ridge_reg, 'sgd_model_leg1.pkl') 

    #Categorical boolean mask
    ohe=OneHotEncoder()
    oh=ohe.fit_transform(timeline_cat)
    joblib.dump(ohe.categories_, 'ohecategories_leg1.pkl') 
    return(svmj)


def Predict(test_path):
    location = test_path
    from sklearn.externals import joblib 
    svmleg1= joblib.load('svm_model_leg1.pkl')  
    ohecatvar1=joblib.load('ohecategories_leg1.pkl') 

    df = pd.read_csv(location)
    df = df.drop(df.columns[[0]], axis=1)

    #Changing dtype
    for col in [stringvar]:
        df[col] = df[col].astype('str')

    for col in [floatvar]:
        df[col] = df[col].astype('float64')   

    for col in [datetimevar]:
        df[col] = df[col].astype('datetime64[ns]')   


    timeline_cat=df[cat]
    timeline_num=df[num]

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn import preprocessing

    num_attribs = list(timeline_num)
    cat_attribs = list(timeline_cat)

    from sklearn.base import BaseEstimator, TransformerMixin
    class DataFrameSelector(BaseEstimator,TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self,X):
            return X[self.attribute_names].values

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer',Imputer(strategy="mean",axis=0)),
        ('std_scaler',StandardScaler()),
    ])

    cat_pipeline1 = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('imputer',SimpleImputer(strategy="constant",fill_value="Unknown", verbose=0, copy=True, add_indicator=True)),
        ('onehotencoderleg', OneHotEncoder(categories=ohecatvar1,sparse=False, handle_unknown='ignore')),

    ])


    #Feature union of different pipelines into full pipeline
    from sklearn.pipeline import FeatureUnion

    full_pipeline1 = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline1),    
    ])


    #leg1 predict df creation
    timeline_prepared = full_pipeline1.fit_transform(df)
    df_predict=svmleg1.predict(timeline_prepared)
    # df_predict=pd.DataFrame(df_predict)
    # df_predict=df_predict.rename(columns={0: "Predicted_Result_Created_to_ApplicationReceived"})
    # df_concat = pd.concat([df,df_predict], axis=1)


    # pd.DataFrame(df_concat).to_csv('Test_Results_Leg1.csv')
    # timeline_prepared.shape
    return (df_predict)
    
def Model_attributes():
    attributes = cat + num
    return (len(attributes),cat,num)
    
def Validate_features_with_database():
    location = test_path    
    df = pd.read_csv(location)
    df = df.drop(df.columns[[0]], axis=1)
    att = num + cat
    result =  all(elem in df for elem in att)
    match = list(set(att)-set(df))
    return(result,match)   

# #Trying regression and prediction on some unseen test data
# attributes = num_attribs + cat_attribs
# dff = pd.read_csv('TEST_SET_SEP30th.csv')
# dff = dff.drop(dff.columns[[0]], axis=1)
# for col in [stringvar]:
#     dff[col] = dff[col].astype('str')
    
# for col in [floatvar]:
#     dff[col] = dff[col].astype('float64')   
    
# for col in [datetimevar]:
#     dff[col] = dff[col].astype('datetime64[ns]')  
# some_data = dff.iloc[20:25]
# some_data_prepared = full_pipeline.transform(some_data)

# #print("Days ", list(some_data))
# print("\n\nPred Forest:", forest_reg.predict(some_data_prepared))
# print("Pred SVM:", svm_reg1.predict(some_data_prepared))
# print("Pred SGD:", sgd_ridge_reg.predict(some_data_prepared))
# some_data.tlmMeterCnt

# s = attr_adder.fit_transform(some_data)
# s = pd.DataFrame(
#     s,
#     columns=list(some_data.columns)+["Days_created_to_application","Days_application_to_design","Days_design_to_inspection","Days_inspection_to_agreement","Days_agreement_to_pto","Days_recieved_to_status","Days_created_to_design","Days_created_to_inspection","Days_created_to_agreement","Days_created_to_pto"],
#     index=some_data.index)

# s = s.dropna(subset=[leg])
# s[leg]
# #s.tlmMeterCnt


# In[7]:


#path = 'CNP_Live_Sep23rd.csv'
#test_path='TEST_SET_SEP30th.csv'
#print(Train(path))
#Predict(test_path)
#Model_attributes()
Validate_features_with_database()


# In[ ]:




