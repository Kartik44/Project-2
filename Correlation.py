#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the packages we'll need during this project

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



#Now we need to read in the data
df = pd.read_csv(r"E:\movies.csv")
df


# In[3]:


df.head()


# In[4]:


# Handling missing data/values 

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,round(pct_missing*100)))


# In[5]:


#checking for any outliers

df.boxplot(column=['gross'])


# In[6]:


df.drop_duplicates()


# In[7]:


# Ordering our Data 

df.sort_values(by=['gross'], inplace=False, ascending=False)


# ### A super quick fact-check

# In[8]:


#scatter plot with budget Vs gross 

plt.scatter(x=df['budget'],y=df['gross'])
plt.title('Budget Vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget')
plt.show


# In[9]:


#Plot budget Vs gross using seaborn

sns.regplot(x="gross", y="budget", data=df , scatter_kws={"color":"red"}, line_kws={"color":"green"})


# In[38]:


sns.stripplot(x="rating", y="gross", data=df)


# In[34]:


sns.regplot(x="score", y="gross", data=df , scatter_kws={"color":"black"}, line_kws={"color":"red"})


# ### Correlation

# In[11]:


# Correlation Matrix between all numeric columns
# using different correlation methods like Pearson,Kendall,Spearman

# we're using Pearson(dy default)

df.corr(method ='pearson')


# In[12]:


df.corr(method ='kendall')


# In[13]:


df.corr(method ='spearman')


# In[14]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[15]:


df_numerised = df

for col_name in df_numerised.columns:
    if(df_numerised[col_name].dtype == 'object'):
        df_numerised[col_name]=df_numerised[col_name].astype('category')
        df_numerised[col_name]=df_numerised[col_name].cat.codes    
        
        #cat.codes is used for converting the variables into numeric values for further analysis

df_numerised


# In[16]:


# Using factorize - this assigns a random numeric value for each unique categorical value

df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[17]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[18]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[19]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# ### We can now take a look at the ones that have a high correlation (> 0.5)
# 

# In[20]:


# We can now take a look at the ones that have a high correlation (> 0.5)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


# ###  Looking at the top 15 companies by gross revenue

# In[21]:


# Looking at the top 15 companies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[22]:


df['Year'] = df['released'].astype(str).str[:4]
df


# In[23]:


df.groupby(['company', 'year'])[["gross"]].sum()


# In[24]:


#highest grossers according to year of release


CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[25]:


CompanyGrossSum = df.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:





# In[ ]:





# In[ ]:




