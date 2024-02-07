#!/usr/bin/env python
# coding: utf-8

# ![Diabetes%20Prediction%20Image.jpg](attachment:Diabetes%20Prediction%20Image.jpg)

# ## ........................................................ Data Loading ....................................................... 

# ### Importing Libraries 

# In[3]:


# Basic Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 

import warnings 
warnings.filterwarnings("ignore")

# Machine Learning Algorithm librarie
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# training model librarie
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# model saving libraries
import pickle
import joblib


# ### Reading CSV file

# In[4]:


rohan_df = pd.read_csv("diabetes.csv")
rohan_df


# ### Exploring Data

# In[5]:


rohan_df.head(6)


# In[6]:


rohan_df.shape


# In[7]:


rohan_df.describe()


# In[8]:


rohan_df.info()


# In[9]:


rohan_df.isnull().sum()


# In[10]:


rohan_df["Outcome"].value_counts()


# In[11]:


rohan_df.groupby("Outcome").mean()


# ## ............................................ Data Visualization and Analysis .........................................

# ### Histogram  

# In[12]:


rohan_df.hist( figsize = (12,12) , layout = (3,3) , sharex = False)
plt.show()


# ### Boxplot  

# In[13]:


rohan_df.plot(kind="box" , figsize = (12,12) , layout = (3,3) , sharex = False , subplots = True)


# ###  Heatmap 

# In[14]:


sns.heatmap( rohan_df.corr() , annot = True , cmap = "terrain" )
plt.show()


# ### Pairplot 

# In[15]:


sns.pairplot(data = rohan_df)
plt.show()


# ## .................................. Preparing the Data for Model ........................................ 

# ### Feature Scaling  

# In[16]:


from sklearn.preprocessing import StandardScaler 
ss = StandardScaler().fit(rohan_df.drop("Outcome" , axis = 1))
X = ss.transform(rohan_df.drop("Outcome" , axis = 1))
y = rohan_df["Outcome"]


# ### Train Test Split 

# In[17]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split ( X , y , test_size = 0.3 )


# In[18]:


print("X_train :", X_train.size)
print("X_test :", X_test.size)
print("y_train :" ,y_train.size)
print("y_test :", y_test.size)


# ### .......................................... Applying Machine Learning Algorithm .......................................... 

# ### Machine Learning Algorithm : LogisticRegression 

# In[19]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
model1 = lr.fit(X_train , y_train)
Prediction1 = model1.predict(X_test)

print( "Testing Accurancy :" , accuracy_score(y_test , Prediction1))


# ### Machine Learning Algorithm : KNeighborsClassifier

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

KNN = KNeighborsClassifier(n_neighbors=3)
model2=KNN.fit(X_train,y_train)
Prediction2 = model2.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,Prediction2))


# ### Machine Learning Algorithm : SVC

# In[21]:


from sklearn.svm import SVC
SVC = SVC()
model3 = SVC.fit(X_train,y_train)
Prediction3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction3))


# ### Machine Learning Algorithm : DecisionTreeClassifier

# In[22]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
model4 = DT.fit(X_train,y_train)
Prediction4 = model4.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction4))


# ### Machine Learning Algorithm : GaussianNB

# In[23]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
model5 = GNB.fit(X_train,y_train)
Prediction5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction5))


# ### Machine Learning Algorithm : RandomForestClassifier 

# In[24]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
model6 = RF.fit(X_train, y_train)
Prediction6  = model6.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction6))


# In[25]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Log-Reg', 'KNN', 'SVC', 'Des-Tree', 'Gaus-NB', 'RandomForest']
accuracy = [ 78.35 , 79.22 , 78.78 , 73.16 , 80.08 , 82.68 ]
ax.bar(langs,accuracy)
plt.show()


# ### The Best Accuracy is given by  RandomForestClassifier is 82.68 . 

# In[26]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , Prediction6)
cm


# In[26]:


sns.heatmap(cm , annot = True , cmap = "BuPu")
plt.show()


# ### precision and recall of the model

# In[27]:


from sklearn.metrics import classification_report
print( classification_report(y_test , Prediction6))


# ### Making a Predictive System 

# In[28]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = ss.transform(input_data_reshaped)
print(std_data)

prediction = model3.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# ### Saving the Model For Future Prediction 

# In[29]:


#Saving Sciketlearn Model
import joblib
joblib.dump ( model6 , "diabetes_prediction_model.pkl")


# In[30]:


import pickle
filename = 'diabetes_prediction_model.sav'
pickle.dump(model6 , open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result,'% Acuuracy')


# In[ ]:




