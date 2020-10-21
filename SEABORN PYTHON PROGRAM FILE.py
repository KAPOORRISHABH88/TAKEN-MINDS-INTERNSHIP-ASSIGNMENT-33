#!/usr/bin/env python
# coding: utf-8

# In[104]:


#IMPORTING PACKAGES
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ### IMPORTING DATA FROM GITHUB LINK

# In[112]:


#USING PANDAS READ METHOD IMPORTING URL
url='https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv'
Data=pd.read_csv(url,sep=',')
Data.head()


# In[114]:


#Checking null or missing values
Data.info()
#As we see Dataframe has zero null or missing values


# In[115]:


#Checking no. of rows and columns through shape argument
Data.shape
#This shows DataFrame has 1704 rows and 6 columns


# **CREATING PIVOT TABLE DATAFRAME**

# In[116]:


Pivot_DF=Data.pivot_table(index='country',columns='year',values='lifeExp')
Pivot_DF.head()


# #### PLOTTING A HEAT MAP USING SEABORN PACKAGE ON TOP OF THIS WIDE TABLE

# In[117]:


plt.figure(figsize=(32,30))
sns.heatmap(data=Pivot_DF,annot=True,fmt='f').get_figure().savefig('HEAT_MAP1.png')
# fmt parameter is used bcoz my values are in float 


# #### PLOTTING A FILTERED  HEAT MAP WITH TOP 11 RECORDS USING CENTER PARAMETER

# In[118]:


plt.figure(figsize=(20,18)) #USED to increase the size of heatmap plot
Pivot_DF1=Pivot_DF.head(11)
sns.heatmap(data=Pivot_DF1,annot=True,fmt='f',center=Pivot_DF1.loc['Australia',1982]).get_figure().savefig('HEAT_MAP2.png')
#center parameter is used to center the colormap of plot


# In[ ]:





# In[ ]:




