from sklearn.externals import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello, World!'


@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    
	df = pd.read_csv("./data/titanic.csv")
	predictors = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	label = 'Survived'
	cabin_fillna = 'NA'
	df.Cabin = df.Cabin.fillna(cabin_fillna)

	df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)

	age_fillna = df_train.Age.mean()
	embarked_fillna = df_train.Embarked.value_counts().index[0]


	df_train.Age = df_train.Age.fillna(df.Age.mean())
	df_train.Embarked = df_train.Embarked.fillna(embarked_fillna)

	df_test.Age = df_test.Age.fillna(df.Age.mean())
	df_test.Embarked = df_test.Embarked.fillna(embarked_fillna)

	le = dict()
	for column in df_train.columns:
	    if df_train[column].dtype == np.object:
	        le[column] = LabelEncoder()
	        df_train[column] = le[column].fit_transform(df_train[column])
	        
	for column in df_test.columns:
	    if df_test[column].dtype == np.object:
	        df_test[column] = le[column].transform(df_test[column])

	model = RandomForestClassifier(n_estimators=25, random_state=42)
	model.fit(X=df_train, y=y_train)
	# y_pred = model.predict(X=df_test)
	# print(confusion_matrix(y_test, y_pred))
	# print(f1_score(y_test, y_pred))
	from sklearn.externals import joblib
	joblib.dump(model, './model/model.pkl') 

	return 'Success'


@app.route('/predict', methods=['POST'])
def predict():
	# data = request.get_json(force=True)
	data = json.loads(request.get_json(s))
	predict_request = [data['Pclass'], data['Sex'], data['Age'], data['SibSp'], data['Parch'], data['Fare'], data['Embarked']]
	predict_request = np.array(predict_request).reshape(1, -1)
	prediction = model.predict(predict_request)
	output = prediction[0]
	return jsonify({'prediction':str(output), 'results': 'Success'})

if __name__ == '__main__':
	model = joblib.load('./model/model.pkl')
	app.run(port=8080)