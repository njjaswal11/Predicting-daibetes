#!/usr/bin/env python
# coding: utf-8

#  # Diabetes prediction
# 
# The goal is to predict whether or not a patient has diabetes based on certain diagnostic parameters included in the kaggle dataset.
# 

# **Importing all the necessary libaries** 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

File= '/kaggle/input/diabetes-healthcare-comprehensive-dataset/health care diabetes.csv'


# In[2]:


df= pd.read_csv(File)


# # Data Cleaning 

# In[3]:


df.head()


# In[4]:


df.describe()


# **Lets check how many null values we have in our data set.**

# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# # Data Manipulation

# In[7]:


df["Outcome"].value_counts()


# In[8]:


df["Outcome"].value_counts().plot(kind = "pie")
plt.show()


# **Because the classes in Outcome are skewed, we will produce fresh samples for the class '1', which is under-represented in our data, using SMOTE.**

# In[9]:


X = df.drop("Outcome", axis = 1)
Y = df["Outcome"]


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[11]:


X_resample, Y_resample = SMOTE(random_state=42).fit_resample(X,Y)


# In[12]:


Y_resample.value_counts().plot(kind = "pie")
plt.show()

Y_resample.value_counts()


# **Now that the data is evenly distributed, we will join this to a new data frame**

# In[13]:


dfn = pd.concat([X_resample,Y_resample],axis = 1)
dfn


# **We will plot scatter plots between all variables**

# In[14]:


plt.figure(figsize=(15,15))
sns.pairplot(dfn,hue="Outcome")
plt.title("Scatter Plot between all the features")
plt.tight_layout()


# **We will plot a heat map to visualize the corrolation between all variables to understand the relationships**

# In[15]:


plt.figure(figsize = (15,15))
sns.heatmap(dfn.corr(),  annot=True)
plt.show()


# In[16]:


dfn.corr()['Outcome'].sort_values()


# # Data modeling

# ***
# We will train our data using several classification models and then evaluate their performance on the test data to accurately predict the desired variable "Outcome" using the following features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeF, and Age.
# 
# The classification models listed below will be used:
# 
# 1. Logistic Regression
# 2. Support Vector Machine
# 3. Decision Tree
# 4. K-Nearest Neighbour 
# ***
# 
# 

# **We will perform Train - Test split on input data**

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[18]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# ## Logistic Regression

# In[19]:


parameters ={"C":np.logspace(-10,50,150),"penalty": ["l1","l2"],"max_iter":[100,400]}
lr=LogisticRegression()
logreg_cv = GridSearchCV(lr, param_grid=parameters, cv=10, verbose=0)


# In[20]:


logreg_cv.fit(X_train,Y_train)


# **We will get the optimal hyperparameters then We will then retrain the models using these optimized hyperparameters**

# In[21]:


logreg_cv.best_params_


# In[22]:


LR= LogisticRegression(C =  4912.190125853841, max_iter =  100)


# In[23]:


LR.fit(X_train,Y_train)


# In[24]:


LR.score(X_test,Y_test)


# **we will plot a heat map to visualize the confusion matrix**

# In[25]:


def plot_cm(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show() 


# In[26]:


yhat=LR.predict(X_test)
plot_cm(Y_test,yhat)


# ## Support Vector Machine

# In[27]:


svm1 = SVC()


# In[28]:


svm1.fit(X_train, Y_train)


# In[29]:


params_svm = {
'C':[1,5, 10, 15, 20, 25],
'gamma':["scale","auto"]
}


# In[30]:


grid_svm = GridSearchCV(svm1,param_grid=params_svm,cv = 10, verbose=0)


# In[31]:


grid_svm.fit(X_train,Y_train)


# In[32]:


grid_svm.best_params_


# In[33]:


svm2 = SVC(C = 5, gamma = "scale", probability=True)


# In[34]:


svm2.fit(X_train,Y_train)


# In[35]:


svm2.score(X_test,Y_test)


# In[36]:


yhat=svm2.predict(X_test)
plot_cm(Y_test,yhat)


# ## Decision Tree

# In[37]:


tree = DecisionTreeClassifier()


# In[38]:


tree.fit(X_train,Y_train)


# In[39]:


parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)]}


# In[40]:


tree_cv = GridSearchCV(tree, param_grid=parameters, cv=10)
tree_cv.fit(X_train, Y_train)


# In[41]:


tree_cv.best_params_


# In[42]:


Tree= DecisionTreeClassifier(criterion='entropy', splitter= 'best', max_depth =4, )


# In[43]:


Tree.fit(X_train,Y_train)


# In[44]:


Tree.score(X_test,Y_test)


# In[45]:


yhat=Tree.predict(X_test)
plot_cm(Y_test,yhat)


# ## K-Nearest Neighbour 

# In[46]:


knn = KNeighborsClassifier()


# In[47]:


knn.fit(X_train, Y_train)


# In[48]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}


# In[49]:


knn = GridSearchCV(knn, param_grid=parameters, cv=10)
knn.fit(X_train, Y_train)


# In[50]:


knn.best_params_


# In[51]:


Knn= KNeighborsClassifier(n_neighbors= 9, algorithm= 'auto', p=1)


# In[52]:


Knn.fit(X_train, Y_train)


# In[53]:


Knn.score(X_test,Y_test)


# In[54]:


yhat=Knn.predict(X_test)
plot_cm(Y_test,yhat)


# **From the models we can see that Support vector machine has the highest accuracy, this will be our final model**

# In[55]:


final_model = svm2


# We will create a report using our final model toget more insight.

# In[56]:


report = classification_report(Y_test,final_model.predict(X_test))

print(report)


# # Dashboard
# we created a dashboard in tableau for more visualization.
# [<div class='tableauPlaceholder' id='viz1692024843495' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DiabetespredictionDataanalysisandML&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DiabetespredictionDataanalysisandML&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DiabetespredictionDataanalysisandML&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1692024843495');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1000px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1000px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1200px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>](http://)
