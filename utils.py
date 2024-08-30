#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sklearn.metrics
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , recall_score, f1_score ,classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from autofeatselect import CorrelationCalculator, FeatureSelector, AutoFeatureSelect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector



df = pd.read_csv("HR_Dataset.csv")
df.head()



df["projects_years"] = df["time_spend_company"] / df["number_project"]
df["month_project"] = df["number_project"] / df["average_montly_hours"]




X = df.drop(columns=["left", "Departments "], axis= 1)
y = df["left"]







# In[11]:


x_train , x_test , y_train , y_test = train_test_split(X, y, test_size=.3 , random_state=66)


# In[12]:


num_cols = [col for col in  x_train.columns 
             if x_train[col].dtype in ['float64', 'int64']]

categ_cols = [col for col in  x_train.columns 
                if x_train[col].dtype not in ['float64', 'int64']]

print('Numerical Columns : \n', num_cols)
print('**'*30)
print('Categorical Columns : \n', categ_cols)


# In[13]:


num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(num_cols)),    
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])

categ_pipeline = Pipeline(steps=[
            ('selector', DataFrameSelector(categ_cols)),    
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OHE', OneHotEncoder(sparse=False))])

total_pipeline = FeatureUnion(transformer_list=[
                                ('num_pip', num_pipeline),
                                ('categ_pipeline', categ_pipeline)])


x_train = total_pipeline.fit_transform(x_train)


# In[14]:


def preprocess_new(X_new):
    return total_pipeline.transform(X_new)


# In[ ]:




