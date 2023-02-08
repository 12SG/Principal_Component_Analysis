#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[2]:


plt.style.use("ggplot")


# In[3]:


from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data,columns=cancer_data.feature_names)


# In[4]:


cancer_data.target_names


# In[5]:


cancer_data.target


# In[6]:


df


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


X = df.values


# In[10]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[11]:


X_scaled.shape


# In[12]:


X_scaled.T.shape


# In[13]:


cov_m = np.cov(X_scaled.T)
cov_m.shape


# In[14]:


cov_m


# In[15]:


eigenvalues,eigenvectors = np.linalg.eig(cov_m)


# In[16]:


eigenvalues


# In[17]:


eigenvalues.shape


# In[18]:


eigenvectors.shape


# In[19]:


eigenvectors = eigenvectors.T


# In[20]:


a = np.array([4,5,3,3,5,5,4,5])
np.cumsum(a)


# In[21]:


eigenvalues = np.cumsum(eigenvalues)
eigenvalues


# In[22]:


13.30499079/30.0528169


# In[23]:


eigenvalues /=eigenvalues.max()
eigenvalues


# In[24]:


0.63243208-0.4427202


# In[25]:


eigenvectors.shape


# In[26]:


p = eigenvectors[0:10,:]
p.shape


# In[27]:


data_new = np.dot(p,X_scaled.T)
data_new.shape


# In[28]:


data_new = data_new.T
data_new.shape


# In[29]:


df_new = pd.DataFrame(data_new,columns=["PC1","PC2","PC3",
                                       "PC4","PC5","PC6","PC7",
                                       "PC8","PC9","PC10"])
df_new.head()


# In[30]:


print("Transformed data:",df_new.shape)
print("Original data:",df.shape)


# In[31]:


plt.figure(figsize=(10,7))

plt.plot(data_new)
plt.xlabel("Observation")
plt.ylabel("Transformed Data")
plt.title("Transformed Data by the principle components(95.15% variablity)",pad = 15)
plt.savefig("plot_manula.png")


# In[32]:


from sklearn.decomposition import PCA


# In[33]:


pca = PCA(n_components=2)
pca_values = pca.fit_transform(X_scaled)


# In[34]:


pca_values


# In[35]:


pca.explained_variance_ratio_


# In[36]:


pca = PCA(n_components = 15)
pca_values = pca.fit_transform(X_scaled)


# In[37]:


pca_values


# In[38]:


var = pca.explained_variance_ratio_
var


# In[39]:


var2 = np.cumsum(np.round(var,decimals=4)*100)
var2


# In[40]:


#Varience plot for PCA components obtained
plt.plot(var2,color="red")
plt.grid()
plt.xlabel("No of PC")


# In[41]:


#Plot between PCA1 and PCA2
x = pca_values[:,0]
y = pca_values[:,1]
#z = pca_values[:,2:3]
plt.scatter(x,y)


# In[42]:


df = pd.DataFrame({"pc1":pca_values[:,0],"pc2":pca_values[:,1]})


# In[43]:


import seaborn as sns
sns.scatterplot(data=df,x='pc1',y='pc2')


# In[ ]:




