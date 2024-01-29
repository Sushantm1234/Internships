#!/usr/bin/env python
# coding: utf-8

# step 1 - Import Library

# In[1]:


import pandas as pd


# step - Import Data

# In[2]:


credit = pd.read_csv('Credit Default.csv')


# In[3]:


credit


# In[4]:


credit.head()


# In[5]:


credit.info()


# In[6]:


credit.describe()


# count of each category

# In[7]:


credit['Default'].value_counts()


# step 3 - define target(y) and features(x)

# In[8]:


credit.columns


# In[10]:


y = credit['Default']


# In[11]:


x = credit.drop(['Default'], axis=1)


# step 4 - train test split

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)


# check shape of train and test sample

# In[14]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# step 5 - select model

# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


model = LogisticRegression()


# step 6 - train or fit model

# In[17]:


model.fit(x_train, y_train)


# In[18]:


model.intercept_


# In[19]:


model.coef_


# step 7 - predict model

# In[20]:


y_pred = model.predict(x_test)


# In[21]:


y_pred


# step 8 - model accuracy

# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[23]:


confusion_matrix(y_test, y_pred)


# In[24]:


accuracy_score(y_test, y_pred)


# In[25]:


print(classification_report(y_test, y_pred))


# In[ ]:




