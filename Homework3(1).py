#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


breast = load_breast_cancer()
breast_data = breast.data
breast_input = pd.DataFrame(breast_data)
breast_labels = breast.target
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data, labels], axis = 1)
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels
breast_dataset['label'].replace(0, 'benign', inplace=True)
breast_dataset['label'].replace(1, 'malignant', inplace=True)
breast_dataset.head()


# In[6]:


X=breast.data
y=breast.target
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.20, train_size = 0.80, random_state = 101)


# In[7]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[8]:


model = GaussianNB()
model.fit(breast.data, breast.target)
print(model)

expected = breast.target
predicted = model.predict(breast.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:




