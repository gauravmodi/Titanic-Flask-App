
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[4]:

df = pd.read_csv("./data/titanic.csv")
predictors = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
label = 'Survived'
cabin_fillna = 'NA'
df.Cabin = df.Cabin.fillna(cabin_fillna)

df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)


# In[5]:

age_fillna = df_train.Age.mean()
embarked_fillna = df_train.Embarked.value_counts().index[0]


df_train.Age = df_train.Age.fillna(df.Age.mean())
df_train.Embarked = df_train.Embarked.fillna(embarked_fillna)

df_test.Age = df_test.Age.fillna(df.Age.mean())
df_test.Embarked = df_test.Embarked.fillna(embarked_fillna)


# In[6]:

le = dict()
for column in df_train.columns:
    if df_train[column].dtype == np.object:
        le[column] = LabelEncoder()
        df_train[column] = le[column].fit_transform(df_train[column])
        
for column in df_test.columns:
    if df_test[column].dtype == np.object:
        df_test[column] = le[column].transform(df_test[column])


# In[8]:

model = RandomForestClassifier(n_estimators=25, random_state=42)
model.fit(X=df_train, y=y_train)
# y_pred = model.predict(X=df_test)
# print(confusion_matrix(y_test, y_pred))
# print(f1_score(y_test, y_pred))
from sklearn.externals import joblib
joblib.dump(model, './model/model.pkl') 


# In[142]:

import requests, json
url = 'http://localhost:8080/predict'
data = {'Pclass':3, 'Sex':1, 'Age':29.699, 'SibSp':1, 'Parch':1, 'Fare':15.24, 'Embarked':0}
# r = requests.post(url, data)
r = requests.post(url, json=json.dumps(data))
r.json()

