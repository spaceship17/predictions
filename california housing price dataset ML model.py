#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('housing.csv')


# In[4]:


data


# In[5]:


data.info()


# In[6]:


data.dropna(inplace =True)


# In[7]:


data.info()


# In[9]:


from sklearn.model_selection import train_test_split

x = data.drop(['median_house_value'], axis =1)
y = data['median_house_value']


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[16]:


train_data = x_train.join(y_train)


# In[17]:


train_data


# In[80]:


train_data.hist(figsize=(15,8))
plt.tight_layout()


# In[20]:


train_data.corr()


# In[22]:


plt.figure(figsize= (15,8))
sns.heatmap(train_data.corr(),annot = True, cmap = "YlGnBu")


# In[24]:


train_data['total_rooms'] = np.log(train_data['total_rooms'] +1 )
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] +1 )
train_data['population'] = np.log(train_data['population'] +1 )
train_data['households'] = np.log(train_data['households'] +1 )                        


# In[81]:


train_data.hist(figsize=(15,8))
plt.tight_layout()


# In[29]:


train_data.ocean_proximity.value_counts()


# In[31]:


pd.get_dummies(train_data.ocean_proximity)


# In[32]:


train_data.join(pd.get_dummies(train_data.ocean_proximity))


# In[34]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis =1)


# In[37]:


train_data.head(5)


# In[38]:


plt.figure(figsize= (15,8))
sns.heatmap(train_data.corr(),annot = True, cmap = "YlGnBu")


# In[82]:


plt.figure(figsize = (15,8))
sns.scatterplot(x = "latitude", y ="longitude", data =train_data, hue="median_house_value", palette = "flare")
plt.tight_layout()


# In[46]:


train_data['bedroom_ratio'] = train_data['total_bedrooms']/train_data['total_rooms']


# In[50]:


train_data['bedroom_ratio'].head(5)


# In[51]:


train_data['household_rooms'] = train_data['total_rooms']/train_data['households']


# In[53]:


train_data['household_rooms'].head(5)


# In[83]:


plt.figure(figsize= (15,8))
sns.heatmap(train_data.corr(),annot = True, cmap = "YlGnBu")
plt.tight_layout()


# In[66]:


from sklearn.linear_model import LinearRegression



x_train, y_train = train_data.drop(['median_house_value'], axis = 1), train_data['median_house_value']


regn = LinearRegression()
regn.fit(x_train, y_train)


# In[60]:


test_data = x_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] +1 )
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] +1 )
test_data['population'] = np.log(test_data['population'] +1 )
test_data['households'] = np.log(test_data['households'] +1 ) 

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis =1)

test_data['bedroom_ratio'] = test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms']/test_data['households']

x_test, y_test = test_data.drop(['median_house_value'], axis = 1), test_data['median_house_value']


# In[61]:


test_data.head(5)


# In[62]:


regn.score(x_test, y_test)


# In[73]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scl = scaler.fit_transform(x_train)

regn.fit(x_train_scl, y_train)


# x_test_scl = scaler.transform(x_test)

# In[74]:


x_test_scl = scaler.transform(x_test)


# In[75]:


regn.score(x_test_scl, y_test)


# In[76]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train, y_train)


# In[77]:


forest.score(x_test, y_test)


# In[78]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train_scl, y_train)


# In[79]:


forest.score(x_test_scl, y_test)

