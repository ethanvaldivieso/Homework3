#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


# In[2]:


breast = load_breast_cancer()
breast_data = breast.data
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


# In[3]:


x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()


# In[4]:


pca_breast = PCA(n_components=2)
principalComponents = pca_breast.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
                                   , columns = ['principal component 1'
                                   , 'principal component 2'])
principalDf.tail()


# In[5]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[6]:


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['benign', 'malignant']
colors = ['b', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[12]:


X=breast.data
y=breast.target
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.20, train_size = 0.80, random_state = 101)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[13]:


model = GaussianNB()
model.fit(X, y)
print(model)

expected = y
predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:




