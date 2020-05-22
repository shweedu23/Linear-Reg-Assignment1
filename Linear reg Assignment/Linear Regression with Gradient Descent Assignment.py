#!/usr/bin/env python
# coding: utf-8

# 201574939
# KUDZAI SIBANDA
# ASSIGNMENT 1 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pymc3 as pm


# In[2]:


os.chdir(r"C:\Users\shwee\OneDrive\Desktop\AML")


# In[3]:


df = pd.read_csv('weight.csv')


# In[4]:


df


# In[5]:


x = df.iloc[:,1]
y = df.iloc[:,2]
plt.scatter(x,y)
plt.show()


# In[6]:


learning_rate = 0.0001
theta_0 = 10
theta_1 = 15

Epochs = 1000
count = 0
n = float(len(x))

def derivative_theta_0(x,y,y_pred,n):
    return (-1/n)*sum(y - y_pred)


def derivative_theta_1 (x,y,y_pred,n):
    return (-1/n)*sum((y - y_pred)*x)

T1 = []
T2 = []
while count<Epochs:
    y_pred = theta_0 + theta_1 * x
    theta_0 = theta_0 - learning_rate * derivative_theta_0(x,y,y_pred,n)
    theta_1 = theta_1 - learning_rate * derivative_theta_1(x,y,y_pred,n)
    T1.append(theta_0)
    T2.append(theta_1)
    count +=1


print(theta_0, theta_1)


# In[7]:


plt.plot(T1) # theta_0 
plt.show()


# In[8]:


plt.plot(T2) # theta_1


# In[13]:


y_pred = theta_0 + theta_1 * x

plt.scatter(x, y) 
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  # regression line
plt.show()


# In[ ]:




