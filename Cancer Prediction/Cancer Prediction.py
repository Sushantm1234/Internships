#!/usr/bin/env python
# coding: utf-8

# Step 1 - Import Library

# In[1]:


import pandas as pd


# Step 2 - Import Data

# In[2]:


cancer = pd.read_csv('Cancer.csv')


# In[3]:


cancer


# In[4]:


cancer.head()


# In[7]:


cancer.info()


# In[6]:


cancer.describe()


# Step 3 - define target(y) and features(y)

# In[10]:


cancer.columns


# In[13]:


y = cancer['diagnosis']


# In[16]:


x = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)


# Step 4 - train test split

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)


# check shape of train and test sample

# In[19]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# step 5 - select model

# In[20]:


from sklearn.linear_model import LogisticRegression


# In[22]:


model = LogisticRegression(max_iter=5000)


# step 6 - train or fit model

# In[23]:


model.fit(x_train, y_train)


# In[25]:


model.intercept_


# In[26]:


model.coef_


# step 7 - predict model

# In[27]:


y_pred = model.predict(x_test)


# In[28]:


y_pred


# step 8 - model accuracy

# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[36]:


confusion_matrix(y_test, y_pred)


# In[38]:


accuracy_score(y_test, y_pred)


# In[40]:


print(classification_report(y_test, y_pred))


# In[ ]:




