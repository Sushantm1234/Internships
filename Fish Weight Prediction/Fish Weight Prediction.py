#!/usr/bin/env python
# coding: utf-8

# step 1 - Import Library

# In[1]:


import pandas as pd


# step 2- Import data 

# In[2]:


fish = pd.read_csv('Fish.csv')


# In[3]:


fish


# In[4]:


fish.head()


# In[5]:


fish.info()


# In[6]:


fish.describe()


# step 3 - define target(y) and features(x)

# In[7]:


fish.columns


# In[10]:


y = fish['Weight']


# In[11]:


x = fish[['Category', 'Height', 'Width', 'Length1', 'Length2', 'Length3']]


# step 4 - train test split

# In[12]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)


# check shape of train and test sample 

# In[15]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# step 5 - select model

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model = LinearRegression()


# step 6 - train or fit model

# In[18]:


model.fit(x_train, y_train)


# In[19]:


model.intercept_


# In[20]:


model.coef_


# step 7 - predict model

# In[21]:


y_pred = model.predict(x_test)


# In[22]:


y_pred


# step 8 - model accuracy

# In[23]:


from sklearn.metrics import mean_absolute_error, r2_score


# In[24]:


mean_absolute_error(y_test, y_pred)


# In[25]:


r2_score(y_test, y_pred)


# In[ ]:




