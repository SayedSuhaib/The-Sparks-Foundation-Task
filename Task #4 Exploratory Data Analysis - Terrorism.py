#!/usr/bin/env python
# coding: utf-8

# # Task 4 - Exploratory Data Analysis - Terrorism

# Name - Sayed Suhaib Iliyas

# Position - Data Science and Business Analytics Intern @ The Sparks Foundation

# Dataset - : https://bit.ly/34SRn3b

# Libraries Importing 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# data importing

# In[2]:


data = pd.read_csv("D:\\Data Science Project\\terrorism/globalterrorismdb_0718dist.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# selecting relevent columns

# In[5]:


data = data[['iyear','imonth','iday','country_txt','region_txt','provstate','city','location','attacktype1_txt','targtype1_txt','target1','gname','weaptype1_txt','weapsubtype1_txt','nkill','nwound']]


# In[6]:


data.head()


# Renaming selected columns

# In[7]:


data.rename(columns={'iyear':'year','imonth':'month','iday':'day','country_txt':'country','region_txt':'region','provstate':'state',
                       'city':'city','attacktype1_txt':'attack_type','targtype1_txt':'target_type','target1':'target','nkill':'killed',
                       'nwound':'wounded','gname':'group',
                       'weaptype1_txt':'weapon_type','weapsubtype1_txt':'weapon_subtype'},inplace=True)


# In[8]:


data.head()


# Attributes with most and least attacks

# In[9]:


print('Year with the most attacks : {}'.format(data['year'].value_counts().idxmax()))
print('Year with the least attacks : {}'.format(data['year'].value_counts().idxmin()))
print('Month with the most attacks : {}'.format(data['month'].value_counts().idxmax()))
print('Country with the most attacks : {}'.format(data['country'].value_counts().idxmax()))
print('Country with the least attacks : {}'.format(data['country'].value_counts().idxmin()))
print('Region with the most attacks : {}'.format(data['region'].value_counts().idxmax()))
print('Region with the least attacks : {}'.format(data['region'].value_counts().idxmin()))
print('State with the most attacks : {}'.format(data['state'].value_counts().idxmax()))
print('State with the least attacks : {}'.format(data['state'].value_counts().idxmin()))
print('City with the most attacks : {}'.format(data['city'].value_counts().idxmax()))
print('city with the least attacks : {}'.format(data['city'].value_counts().idxmin()))
print('Location with the most attacks : {}'.format(data['location'].value_counts().idxmax()))
print('Location with the least attacks : {}'.format(data['location'].value_counts().idxmin()))
print('Most attack_type : {}'.format(data['attack_type'].value_counts().idxmax()))
print('Least attack_type : {}'.format(data['attack_type'].value_counts().idxmin()))
print('Group with most attacks : {}'.format(data['group'].value_counts().idxmax()))
print('Group with least attacks : {}'.format(data['group'].value_counts().idxmin()))
print('Most used weapon type : {}'.format(data['weapon_type'].value_counts().idxmax()))
print('Least used weapon type : {}'.format(data['weapon_type'].value_counts().idxmin()))
print('Most used weapon subtype : {}'.format(data['weapon_subtype'].value_counts().idxmax()))
print('Least used weapon subtype : {}'.format(data['weapon_subtype'].value_counts().idxmin()))


# Attacks per year

# In[10]:


data['year'].shape


# In[11]:


data['year']


# In[12]:


data['year'].dtype


# In[13]:


data['year'].value_counts()


# In[14]:


x = data['year'].unique()


# In[15]:


x


# In[16]:


y = data['year'].value_counts().sort_index()


# In[17]:


y


# Attacks per year Graph

# In[18]:


plt.figure(figsize = (15,10))
sns.barplot(x,y)
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attack_of_Years')
plt.show()


# Year with most attacks = 2014

# Year with least attacks = 1971

# region vs year attack visualization

# In[19]:


pd.crosstab(data.year, data.region).plot(kind='area',figsize=(15,6))
plt.title('Terrorist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()


# Attacks vs region bar plot

# In[20]:


plt.figure(figsize = (15,10))
sns.barplot(data['region'].value_counts()[:15].index,data['region'].value_counts()[:15].values)
plt.xticks(rotation = 45)
plt.xlabel('Region')
plt.ylabel('Number of Attacks each year')
plt.title('Top Affected Regions')
plt.show()


# Region with most attacks = Middle East & North Africa

# Region with least attacks = Australasia & Oceania

# In[ ]:





# In[21]:


data['country'].value_counts()[:15].index


# In[22]:


data['country'].value_counts()[:15].values


# country vs number of attacks per year bar plot

# In[23]:


plt.figure(figsize = (15,10))
sns.barplot(data['country'].value_counts()[:15].index,data['country'].value_counts()[:15].values)
plt.xticks(rotation = 45)
plt.xlabel('Country')
plt.ylabel('Number of Attacks each year')
plt.title('Top Affected Countries')
plt.show()


# Country with most attacks = Iraq

# Country with least attacks = Spain

# In[24]:


data['wounded'] = data['wounded'].fillna(0).astype(int)
data['killed'] = data['killed'].fillna(0).astype(int)
data['casuality'] = data['wounded'] + data['killed']


# In[25]:


data['casuality']


# In[26]:


data1 = data.sort_values(by = 'casuality' ,ascending = False)[:50]


# In[27]:


data1


# casuality table 

# In[28]:


casuality_table = data1.pivot_table(index = 'country',columns = 'year',values = 'casuality')
casuality_table.fillna(0,inplace = True)
casuality_table


# In[29]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=casuality_table.values, x=casuality_table.columns, y=casuality_table.index, colorscale=colorscale)
hm = [heatmap]
layout = go.Layout(
    title='Top 40 Worst Terror Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data = hm, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[30]:


killdata = data.loc[:,'killed']
killdata


# In[31]:


AttackType = data.loc[:,'attack_type']
AttackType


# In[32]:


KillTypeData = pd.concat([AttackType, killdata], axis=1)


# In[33]:


KillTypeData


# In[34]:


killtype = KillTypeData.pivot_table(columns='attack_type', values='killed', aggfunc='sum')
killtype


# In[35]:


killtype.value_counts()


# Pi Chart of attack types

# In[36]:


fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(aspect="equal"))
label = ["Armed Assault","Assassination","Bombing/Explosion","Facility/Infrastructure Attack","Hijacking","Hostage Taking (Barricade Incident)","Hostage Taking (Kidnapping)","Unarmed Assault","Unknown"]
y = np.array([160297,24920,157321,3642,3718,4478,24231,880,32381])
plt.pie(y,labels=label, startangle=90, autopct='%1.1f%%')
plt.legend()
plt.show() 


# most used attack type = Armed Assault (38.9%)

# least used atack type = Unarmed Assault(0.2%)

# In[37]:


countrydata = data.loc[:,'country']
countrydata


# In[38]:


killdata = data.loc[:,'killed']
killdata


# In[39]:


countrykill = pd.concat([countrydata, killdata], axis=1)
countrykill


# In[40]:


CountryKill = countrykill.pivot_table(columns = 'country', values = 'killed', aggfunc='sum')
CountryKill


# In[41]:


labels1 = CountryKill.columns.tolist()
labels1


# In[42]:


l1 = labels1[:50]
l1


# In[43]:


x1 = np.arange(len(l1))
x1


# In[44]:


transpoze = CountryKill.T
transpoze


# In[45]:


y1 = transpoze.values.tolist()
y1 = y1[:50]
y1
y1 = [int(i[0]) for i in y1]
y1


# country vs kills bar plot

# In[46]:


plt.figure(figsize = (20,20))
sns.barplot(l1,y1)
plt.xticks(rotation = 45)
plt.xlabel('Country')
plt.ylabel('Number of Attacks each year')
plt.title('Top Affected Countries')
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.title('Number of people killed by countries', fontsize = 20)
plt.show()


# In[ ]:




