#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/imdb (1000 movies) in june 2022.csv')
df.head(3)


# In[ ]:


cols=['Ranking', 'Movie_Name', 'Year', 'Certificate', 'Runtime(Minutes)', 'Genre', 'Rating', 'Metascore', 'About',
      'Director', 'Actor_1', 'Actor_2', 'Actor_3', 'Actor_4', 'Votes', 'Gross_Collection(Million)']
df.columns=cols
df = df.drop(columns=['Certificate','About'])
df.sample(5)


# In[ ]:


df=df[(df['Gross_Collection(Million)'].notnull())&(df['Gross_Collection(Million)']!='$0.00M')]


# In[1]:


from sklearn.linear_model import LinearRegression 
                
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(30,24), dpi=60, nrows=2, ncols=2)

x1=df['Runtime(Minutes)']
x2=df['Votes']
x3=df['Gross_Collection(Million)']
x4=df['Metascore']
y=df['Rating']


sns.regplot(x=x1, y=y, ax=ax1, scatter_kws={'alpha':0.28},color='mediumpurple')
sns.regplot(x=x2, y=y, ax=ax2, scatter_kws={'alpha':0.28},color='blue')
sns.regplot(x=x3, y=y, ax=ax3, scatter_kws={'alpha':0.28},color='gold')
sns.regplot(x=x4, y=y, ax=ax4, scatter_kws={'alpha':0.28},color='greenyellow')

ax1.set_title('Rating VS Runtime in Minutes', fontsize=25, fontweight ='heavy',color='mediumpurple')
ax1.set_xticks(range(60,240,20))
ax1.set_xlabel('Runtime (Minutes)',color='black')

ax2.set_title('Rating VS No. of Votes', fontsize=25, fontweight ='heavy',color='blue')
ax2.set_xticks(range(0,2750000,200000))
ax2.set_xlabel('Votes (Million)')

ax3.set_title('Rating VS Gross Collection (Million)', fontsize=25, fontweight ='heavy',color='gold')
ax3.set_xticks(range(0,1000,100))

ax4.set_title('Rating VS Metascore', fontsize=25, fontweight ='heavy',color='greenyellow')

# Calculate r^2
lin_reg_1=LinearRegression().fit(df[['Runtime(Minutes)']], y)
lin_reg_2=LinearRegression().fit(df[['Votes']], y)
lin_reg_3=LinearRegression().fit(df[['Gross_Collection(Million)']], y)
lin_reg_4=LinearRegression().fit(df[['Metascore']].notnull(), y)

r2_1=lin_reg_1.score(df[['Runtime(Minutes)']], y)
r2_2=lin_reg_2.score(df[['Votes']], y)
r2_3=lin_reg_3.score(df[['Gross_Collection(Million)']], y)
r2_4=lin_reg_4.score(df[['Metascore']].notnull(), y)

print(f'r^2 of Rating VS Runtime in Minutes = {round(r2_1,3)}.')
print(f'r^2 of Rating VS No. of Votes = {round(r2_2,3)}.')
print(f'r^2 of Rating VS Gross_Collection(Million) = {round(r2_3,3)}.')
print(f'r^2 of Rating VS Metascore = {round(r2_4,3)}.')

sns.despine()
plt.show()


# In[ ]:




