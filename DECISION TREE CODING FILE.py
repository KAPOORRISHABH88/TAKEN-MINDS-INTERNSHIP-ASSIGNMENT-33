#!/usr/bin/env python
# coding: utf-8

# **IMPORT REQUIRED DEPENDENCIES(PACKAGES)**

# In[55]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


# ***NOW WE IMPORT IRIS DATASET AS PER OUR PROBLEM TO SOLVE***

# In[43]:


iris_df=pd.read_csv("C:/Users/Kapporz/Documents/Taken minds/IRIS DATASET ASSIGNMENT#4/Iris.csv",sep=',')


# In[44]:


iris_df.head()
#USING HEAD COMMAND TO SHOW FIRST FIVE ROWS


# In[45]:


#INSPECT NO. OF ROWS & COLUMNS
iris_df.shape


# ### CREATING INPUT VARIABLE x AND TARGET VARIABLE y

# In[46]:


X=iris_df.values[:,0:5]


# ### AS TARGET VARIABLE IS SPECIES WHICH WE HAVE TO PREDICT FOR NEW DATA AND TEST ITS ACCURACY

# In[47]:


Y=iris_df.values[:,5]


# **SPLITTING DATA INTO TEST AND TRAIN DATA**

# In[49]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=1)


# #### CALLING DECISION TREE CLASSIFIER FUNCTION

# In[50]:


DT_CLASS=DecisionTreeClassifier(criterion='entropy',random_state=1)


# **NOW BUILDING THE MODEL ON TRAINING DATA AS**

# In[51]:


DT_CLASS.fit(x_train,y_train)


# #### Predicting Target Variable

# In[52]:


y_predict=DT_CLASS.predict(x_test)
y_predict


# In[63]:


compare=pd.DataFrame({'y_predict':y_predict,'y_test':y_test})


# In[68]:


compare.head(20)


# In[65]:


#CHECKING ACCURACY
print('Accuracy in decision tree model is ')
accuracy_score(y_test,y_predict)*100


# ### OR by checking accuracy through alternate method of score

# In[66]:


DT_CLASS.score(x_test,y_test)


# In[ ]:




