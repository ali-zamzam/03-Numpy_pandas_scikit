#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# url = 'https://public.opendatasoft.com/explore/dataset/population-francaise-communes/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B'
# data = pd.read_csv(url,sep=";")
# data.head(5)


# In[3]:


data1 = pd.read_csv('Departements.csv', sep=';')
data1.head(5)


# In[4]:


data1.columns=['Code département','département','nb arrondissement',' nb canton','nb commune','Population municipale','Population totale','null']
data1


# In[5]:


data11= data1.copy()
data11.head(1)


# In[6]:


data11.drop(data11.loc[:,"nb arrondissement":"Population municipale"],1,inplace=True)
data11


# In[7]:


del data11["null"]
data11


# In[8]:


data_1_n = data11.copy()
data_1_n


# In[9]:


data_1_n.sort_values("Code département",ascending=0)


# In[10]:


data_1_n["Population totale"].sum()


# <h1>--------------------------------------------------------------------------------------------------------</h1>

# In[11]:


data2 = pd.read_csv('population-francaise-communes.csv', sep=';')
data2.head(1)


# In[12]:


data2[["Code région","Nom de la région","Code département","Nom de la commune","Population totale"]]


# In[13]:


data3 = data2.copy()
data3.head(1)


# In[14]:


# data_1 = data3.drop(labels=iloc["Code arrondissement départemental", "Code canton", "Code commune","Population comptée à part","Population municipale"], axis='columns')
# data_1
# data_1 = data3.drop(labels=iloc[2:5])
# data_1
# data3.drop(data3.iloc[:,3:5],1,inplace=True)
data3.drop(data3.loc[:,"Code arrondissement départemental":"Code commune"],1,inplace=True)


# In[15]:


data3.head(1)


# In[16]:


data_n = data3.copy()
data_n.head(1)


# In[17]:


data_n.drop(data3.loc[:,"Année recensement":"EPCI"],1,inplace=True)


# In[18]:


data_n.head(1)


# In[19]:


data_n.drop(data_n.loc[:,"Population municipale":"Population comptée à part"],1,inplace=True)
data_n.head(1)


# In[20]:


data_ne = data_n.copy()
data_ne.head(1)


# In[21]:


data_ne


# In[22]:


data_ne.isnull()


# In[23]:


data_ne.isnull().value_counts()


# In[24]:


data_new = data_ne.fillna(0)
data_new


# In[25]:


data_new.isnull().value_counts()


# In[26]:


data_new.duplicated()


# In[27]:


data_new1 = data_new.drop_duplicates(['Nom de la commune'])
data_new1


# In[28]:


data_new1["Population totale"].sum()


# In[29]:


data_new1.reset_index(inplace=True)
data_new1


# In[72]:


data_new1.groupby('Code région')['Population totale'].nunique().plot(kind='bar')
plt.show()


# In[73]:


data_1_n


# In[88]:


data_f = data_1_n.merge(data_new1, how="inner", on="Code département")
data_f


# In[90]:


data_f.reset_index(inplace=True)
data_f


# In[92]:


del data_f["Population totale_x"]
data_f


# In[93]:


del data_f["level_0"]
data_f


# In[94]:


del data_f["index"]
data_f


# In[95]:


data_f.rename(columns={'Population totale_y': 'Population totale'}, inplace=True)
data_f


# In[96]:


data_final = data_f.copy()
data_final


# In[97]:


data_final.isnull().value_counts()


# In[98]:


data_final["Population totale"].sum()


# In[100]:


data_final.groupby('département')['Population totale'].nunique().plot(kind='bar')
plt.show()


# In[ ]:




