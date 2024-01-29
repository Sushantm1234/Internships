#!/usr/bin/env python
# coding: utf-8

# Step 1 - Import Library

# In[1]:


import pandas as pd


# Step 2 - Import Data

# In[2]:


ice = pd.read_csv('Ice Cream.csv')


# In[3]:


ice


# In[4]:


ice.head()


# Step 3 - Define target(y) and features(x)

# In[5]:


ice.columns


# In[6]:


y = ice['Revenue']


# In[7]:


x = ice[['Temperature']]


# Step 4 - Train Test Split

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)


# check shape of the train and test sample

# In[10]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# step 5 - Select Model

# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


model = LinearRegression()


# step 6 - train or fit model

# In[13]:


model.fit(x_train, y_train)


# In[14]:


model.intercept_


# In[15]:


model.coef_


# step 7 - predict model

# In[16]:


y_pred = model.predict(x_test)


# In[17]:


y_pred


# step 8 - model accuracy

# In[19]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[20]:


mean_absolute_error(y_test, y_pred)


# In[21]:


mean_absolute_percentage_error(y_test, y_pred)


# In[22]:


mean_squared_error(y_test, y_pred)


# In[ ]:




