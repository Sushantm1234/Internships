#!/usr/bin/env python
# coding: utf-8

# Step 1 - Import Libraries

# In[19]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Step 2 - Import Data

# In[5]:


data = pd.read_csv('Admission Chance.csv')


# In[6]:


data


# In[7]:


data.head()


# In[8]:


data.describe()


# In[9]:


data.info()


# step 3 - Define target (y) and features (x)

# In[12]:


data.columns


# In[15]:


y = data['Chance of Admit ']


# In[16]:


x = data.drop(['Serial No','Chance of Admit '], axis=1)


# In[17]:


x.columns


# step 4 - Train Test Split

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)


# check shape of train and test sample

# In[21]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[22]:


model = LinearRegression()


# In[24]:


model.fit(x_train, y_train)


# In[25]:


model.intercept_


# In[26]:


model.coef_


# step 7 - Predict Model

# In[27]:


y_pred = model.predict(x_test)


# In[28]:


y_pred


# step 8 - Model Accuracy

# In[29]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[30]:


mean_absolute_error(y_test, y_pred)


# In[31]:


mean_absolute_percentage_error(y_test, y_pred)


# In[32]:


mean_squared_error(y_test, y_pred)


# In[ ]:




