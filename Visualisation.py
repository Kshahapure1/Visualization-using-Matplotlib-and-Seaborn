# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:07:28 2021

@author: HP
"""

# Chapter 11 : EDA and Vidualisation
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

cust=pd.read_csv('TelicomCust2021.csv')
print(cust)

print(cust.head(10))

# data types 
cust.dtypes

#dimensions
cust.shape

#Columns
cust.columns

#Information of te data
cust.info()

#Nulls
cust.isnull().sum()

#suppose there are nulls in dataset, what will you do?
#gender has nulls and  #talktime has nulls


#split the columns into numeric columns and factor columns

#Numeric
nc =list(cust.select_dtypes(exclude='object').columns.values)
nc

#Catogerical COLUMNS

#Factor
fc = list(cust.select_dtypes(include='object').columns.values)
fc

#remove the columns that are not catoheries

fc.remove('name')
fc.remove('email')
fc

#Analysis on categorie columns 
    #no. of levels
    #Duplicates values
    #spaces/spelling errors

#EDA for factors
#print the unique values from fc


for c in fc:
    print("unique values for the categorical columns ", c)
    print(cust[c].unique())
    print('---')


# EDA on numeric columns

#pivot table
# average talktime of customers based on city
np.round(pd.pivot_table(cust,"talktime",["city"],aggfunc=np.mean),1)


#total complaints based on plan type
pd.pivot_table(cust,"compl",["plan"],aggfunc=np.sum)

#Combine the aggrigate functions(Total and average in the same output)
#Total and average complaints on plan type

pd.pivot_table(cust,"compl",["plan"],aggfunc=[np.sum,np.mean])

# total cmplaints based on plan typesand city
pd.pivot_table(cust,"compl",["plan","city"],aggfunc=[np.sum])

#-----------------
# Charts 
#-----------------

#1) Line chart
#age vs internet usage
sns.lineplot(cust.age,cust.netuse,color = "red",ci = None)
plt.title("Age vs Internet Use")
#plt.plot(cust.age,cust.netuse,color = "red")

#2) Scatter plot
sns.scatterplot(cust.age,cust.netuse,color = "blue")
plt.title("Age vs Interne Usage")


#3) Count plot
#Count the number of customers by city

sns.countplot(x = cust.city,color = 'cyan')

#Horizonta plot
sns.countplot(y=cust.city,color = "Brown")
plt.title("Distribution of cuatomers by city")

#4) side by side chart(cross tab)
#plot distribution of custoemrs by the plan and city

pd.crosstab(cust["plan"],cust["city"]).plot(kind="bar")
plt.title("Plan - City")
plt.xlable("Mobile plan")
plt.ylable("Count")

#5) Bar chart
#City vs Complaints

sns.barplot(x=cust.city,y=cust.compl,color = "Green",ci = None)
plt.title("City vs Complaingts")

# Horizontal 
sns.barplot(x=cust.compl,y=cust.city,color = "Green",ci = None)
plt.title("City vs Complaingts")

#Dodge bar
#City wise registered complaints, group by gender
sns.barplot(x=cust.city,y=cust.compl,hue=cust.gender,ci=None)
plt.title("City wise registered complaints , Group by gender")

#####

# Chart specific to Numeric data

#1) histogram (use to identify the distribution )
sns.distplot(cust.age,bins=20,color = "Green")

#plot only the curve(KDE = Kennal density estimation)
sns.kdeplot(cust.age)

#histogram for all te colunmns 
ROWS = len(nc)/2
COLS = 2
POS = 1

fig = plt.figure() # Outer most plot
#plot each columns as subplot means withing the main plot
for c in nc:
    fig.add_subplot(ROWS,COLS,POS)
    sns.distplot(cust[c],bins=20,color = "Green")
    POS +=1
    
#2) Box Plot (To identify distribution and outliers)
sns.boxplot(cust.netuse,color = "Yellow")

#verticl format
sns.boxplot(cust.netuse,color="yellow",orient="v")    


# Box pot group by categorical column
#netusage group by city

sns.boxplot(x=cust.netuse,y=cust.city)
plt.title("Net usage city wise")

#Vertical format
sns.boxplot(x=cust.city,y=cust.netuse)

#A1) Use loop to boxplot all the numeric data in a single frame

 

#Correlation Matrix using Heatmap
print(nc)

#take only the relivent columns for the correlation matrix
nc.remove("custid")


cor = cust[nc].corr()
cor
#take only the lower triangle for correlation values and fill the upper triangle with zero
cor = np.tril(cor)
cor
sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,yticklabels=nc,square=False,annot=True,linewidths=1)
plt.title("Correlation Mtrix")

#pie chart
#to plot the % of distribution

data = round(cust.groupby('city').compl.mean(),1)
data

data.plot.pie(autopct="%.1f%%")

import numpy as np

# put null values in 10 random records

gen_ndx = np.random.randint(0,cust.shape[0],10)
gen_ndx_value = cust.gender[gen_ndx]
cust.gender[gen_ndx] = None
cust.isnull().sum()

#How to impute nulls in gender?
cust.gender[gen_ndx]

#count the number of records for he gender column
cust.gender.value_counts()
cust.gender[gen_ndx] = None

for n in range(0,len(gen_ndx)):
    if n%2 == 0:
        print(n, "F")
        cust.gender[cust.index==gen_ndx[n]] = "F"
    else:
        print(n, 'M')
        cust.gender[cust.index==gen_ndx[n]] = "M"
        
#check the updated records
cust.gender[gen_ndx]
cols = ['gender','city','plan']
cust[cols][cust.gender.isnull()]

# Randomly put 0 value in "age"

age_ndx = np.random.randint(0,cust.shape[0],25)
age_ndx_value = cust.age[age_ndx]
cust.age[age_ndx] = 0
cust.age[age_ndx]
cust.age[age_ndx] = np.random.randint(5,35)
cust[cust==0].count()

cust.age[cust.age>=18].describe()

for i in range(0,len(age_ndx)):
    print(i)
    cust.age[cust.index==age_ndx[i]] = np.random.randint(18,89)
    
#check the updated records
cust.age[age_ndx]


# randomly put outliers to thr "netuse
netuse_ndx = np.random.randint(0,cust.shape[0],10)

netuse_ndx_value = cust.netuse[netuse_ndx]
netuse_ndx

cust.netuse[netuse_ndx] # to check the plan

for i in range(0,len(netuse_ndx)):
    print(i)
    cust.netuse[cust.index==netuse_ndx[i]] = np.random.randint(15,50)












































