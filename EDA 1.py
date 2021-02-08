#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis(EDA)
# 

# In[ ]:


# EDA is nothing but a data exploration technique to understand the various aspects of data.
# Objective of EDA:
#To filter the data from redundancies
#To understand the relationships between the variables

# Steps in EDA
1. Understand the Data
2. Clean the Data
3. Analysis of relationship between variables
# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the titanic dataset
titanic=sns.load_dataset('titanic')


# In[3]:


titanic.head(10)


# In[4]:


# count the number of rows and columns in the dataset
titanic.shape


# In[5]:


# Get some statistics on columns with numerical values
titanic.describe()


# In[6]:


# for lables of the columns
titanic.columns


# In[7]:


# for missing values
titanic.isnull()


# In[8]:


titanic.isnull().sum()


# In[9]:


sns.countplot(x='sex',data=titanic)


# In[10]:


# visualization of missing values
titanic.corr()
sns.heatmap(titanic.corr())


# In[11]:


sns.heatmap(titanic.isnull(),xticklabels=True, yticklabels=False, cbar=False,cmap='viridis')


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='survived',hue='sex',data=titanic)


# In[13]:


plt.figure(figsize=(12,7))
sns.boxplot(x='pclass', y='age',data=titanic)


# In[14]:


# to drop the Nan values
titanic['sibsp'].dropna()


# In[15]:


sns.distplot(titanic['sibsp'].dropna())


# In[16]:


# replacing the missing values
def impute_age(cols):
    Age = cols[0]
    pclass = cols[1]
    if pd.isnull(Age):
        if pclass == 1:
             return 37
        elif pclass == 2:
             return 29
        else:
            return 24
    else:
        return Age


# In[17]:


# call the .apply function
titanic['age'] = titanic[['age','pclass']].apply(impute_age,axis=1)


# In[18]:


# again check the heatmap function
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# missing values of age column replaced with mean values with help of function written above


# In[19]:


# converting to catogrical variables
pd.get_dummies(titanic['embarked'],drop_first=True).head()


# In[20]:


embark=pd.get_dummies(titanic['embarked'],drop_first=True)
sex=pd.get_dummies(titanic['sex'],drop_first=True)


# In[21]:


# dropoing the redundant varibles
titanic.drop(['sex','embarked'],axis=1, inplace=True)
titanic.head()


# In[22]:


#
titanic=pd.concat([titanic,sex,embark],axis=1)
titanic.head()


# In[23]:


titanic=sns.load_dataset('titanic')


# In[24]:


# Get a count of number of survivors
titanic['survived'].value_counts()


# In[25]:


# Visualize the number of survivors
sns.countplot(titanic['survived'] )


# In[26]:


# visualize the count of survivors for multiple columns (who,sex,pclass,sibsp,parch,embarked)
cols=['who','sex','pclass','sibsp','parch','embarked']
n_rows=2
n_cols=3
# the subplot grid and figure size of each graph
fig, axs=plt.subplots(n_rows, n_cols, figsize = (n_cols * 3.2, n_rows * 3.2))
for r in range(0, n_rows):
    for c in range(0,n_cols):
        i=r*n_cols + c
        ax = axs[r][c]
        sns.countplot(titanic[cols[i]],hue=titanic['survived'],ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='survived',loc='upper right')
plt.tight_layout()


# In[27]:


# look at the survival rate by 'sex'
titanic.groupby('sex')[['survived']].mean()


# In[28]:


# look at the survival rate by sex and class
titanic.pivot_table('survived',index='sex',columns='class')


# In[29]:


# look at the survival rate by sex and class visually
titanic.pivot_table('survived',index='sex',columns='class').plot()


# In[30]:


# plot yhe survival rate of each class
sns.barplot(x='class',y='survived',data=titanic)


# In[31]:


# look at the survival rate by sex,age and class
age=pd.cut(titanic['age'],[0,18,80])
titanic.pivot_table('survived',['sex',age],'class')


# In[32]:


# plot the prices paid by each class
plt.scatter(titanic['fare'],titanic['class'],color='purple',label='Passenger paid')
plt.ylabel('class')
plt.xlabel('price')
plt.title('prices paid by each class')
plt.legend()


# In[33]:


# count the empty values in each column
titanic.isna().sum()


# In[34]:


# look at all the values in each column and get a count
for val in titanic:
    print(titanic[val].value_counts())
    print()


# In[35]:


# drop the columns
titanic=titanic.drop(['deck','embark_town','alive','class','who','alone','adult_male'],axis=1)
# Remove the rows with missing values
titanic=titanic.dropna(subset = ['embarked','age'])


# In[36]:


# Count the New number of rows and columns in the data set
titanic.shape


# In[37]:


# look at the data types
titanic.dtypes


# In[38]:


# in the above dataset every value is numerical except sex and embarked
# need to transform the sex and embarked values into numericals for analysis


# In[39]:


# print the unique values in the columns
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[40]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Encode the sex column
titanic.iloc[:,2] = labelencoder.fit_transform( titanic.iloc[:,2].values)
# Encod the embarked column
titanic.iloc[:,7] = labelencoder.fit_transform( titanic.iloc[:,7].values)


# In[41]:


# print the unique values in the columns after transformation
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# In[42]:


titanic.dtypes


# In[43]:


# split the data into independent 'x' and dependent 'y' varibles


# In[44]:


x=titanic.iloc[:,1:8].values
y=titanic.iloc[:,0].values


# In[45]:


# split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[46]:


# scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform (x_train)
x_test = sc.fit_transform (x_test)


# In[47]:


# Create a function with various machine learning models
def models(x_train, y_train):
    
    # Use Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(x_train, y_train)
    
    # Use KNeighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier ()
    knn.fit(x_train, y_train)
    
    # Use SVC (linear kernel)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel= 'linear',random_state = 0)
    svc_lin.fit(x_train, y_train)
    
    
    # Use SVC (RBF kernel)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel= 'rbf',random_state = 0)
    svc_rbf.fit(x_train, y_train)
    
    # Use GausssianNB
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)
    
    # Use Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier (criterion = 'entropy', random_state=0)
    tree.fit(x_train, y_train)
    
    # Use the RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=0)
    forest.fit(x_train, y_train)
    
    # print the training accuracy for each model
    print ('[0]Logistic Regression Training Accuracy:', log.score(x_train, y_train))
    print ('[1]K Neighbors Training Accuracy:', knn.score(x_train, y_train))
    print ('[2]SVC Linear Training Accuracy:', svc_lin.score(x_train, y_train))
    print ('[3]SVC RBF Training Accuracy:', svc_rbf.score(x_train, y_train))
    print ('[4]Gaussian NB Training Accuracy:', gauss.score(x_train, y_train))
    print ('[5]Decision TreeTraining Accuracy:', tree.score(x_train, y_train))
    print ('[6]Random Forest Training Accuracy:', forest.score(x_train, y_train))
    
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest
    
    


# In[48]:


# Get and train all of the model
model = models(x_train, y_train)


# In[49]:


# Show the confusion matrix and accuracy for all the models on the test data
from sklearn.metrics import confusion_matrix
for i in range (len(model)):
    cm=confusion_matrix(y_test, model[i].predict(x_test))
    
    # Extract TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(y_test, model[i].predict(x_test)).ravel()
    test_score = (TP+TN)/(TP+TN+FN+FP)
    print(cm)
    print('Model[{}]Testing Accuracy ="{}"'.format(i, test_score) )
    
    print()


# In[50]:


# Get feature importance
forest = model[6]
importances =pd.DataFrame({'feature':titanic.iloc[:,1:8].columns, 'importance': np.round(forest.feature_importances_, 3)})
importances =importances.sort_values('importance', ascending =False).set_index('feature')
importances


# In[51]:


# Visualize the importances
importances.plot.bar()


# In[52]:


# print the prediction of the random forest classifier
pred=model[6].predict(x_test)
print (pred)

print()

# print the actual values
print(y_test)


# In[53]:


#pclass        int64, 
#sex           int32
#age         float64
#sibsp         int64
#parch         int64
#fare        float64
#embarked      int32


# In[54]:


# My survival
my_survival = [[1,0,21,8,6,350,0]]
# Scaling my survival
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
my_survival_scaled = sc.fit_transform(my_survival)

# print prediction of my survival using random forest classifier
pred=model[6].predict(my_survival_scaled)
print(pred)

if pred == 0:
    print('Oh no you did not make it')
else:
    print ('Nice! you have survived')


# In[ ]:





# In[ ]:




