#!/usr/bin/env python
# coding: utf-8

# step 1 - Import Library

# In[1]:


import pandas as pd


# step - Import data

# In[2]:


purchase = pd.read_csv('Customer Purchase.csv')


# In[3]:


purchase


# In[4]:


purchase.info()


# In[5]:


purchase.describe()


# In[6]:


purchase.head()


# step 3 - define target(y) and features(x)

# In[7]:


purchase.columns


# In[8]:


y = purchase['Purchased']


# In[9]:


x = purchase.drop(['Purchased', 'Customer ID'], axis=1)


# encoding catogorical variable

# In[10]:


x.replace({'Review':{'Poor':0, 'Average':1, 'Good':2}},inplace=True)
x.replace({'Education':{'School':0, 'UG':1, 'PG':2}},inplace=True)
x.replace({'Gender':{'Male':0, 'Female':1}},inplace=True)


# display first 5 rows

# In[11]:


x.head()


# step 4 - train test split

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2529)


# check shape of train and test sample

# In[14]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# step 5 - select model

# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


model = RandomForestClassifier()


# step 6 - train or fit model

# In[18]:


model.fit(x_train, y_train)


# step 7 - predict model

# In[19]:


y_pred = model.predict(x_test)


# In[20]:


y_pred


# step 8 - model accuracy

# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[24]:


confusion_matrix(y_test, y_pred)


# In[25]:


accuracy_score(y_test, y_pred)


# In[26]:


print(classification_report(y_test, y_pred))


# In[ ]:




