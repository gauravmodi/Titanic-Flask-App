# Flask app for Titanic: Machine Learning from Disaster
**- Gaurav Modi**<br>
_(last edit: 04/15/2018)_
<p>This is a flask app to predict whether a particular passenger would survive. Along with the label, it shows the Survival probability too.

## Packages Used:
```
+ Json
+ Pandas
+ Numpy
+ Flask
+ WTForms
+ Scikit-Learn
```

## Steps:
### Data preperation and model building
+ Reading data
+ Splitting the data into training and testing data
+ Imputing missing values from training dataset.
+ Using values imputed from training data to fill missing values in testing dataset to avoid data leakage.
+ Encoding Categorical predictors using sklearn LabelEncoder
+ Fitting RandomForestClassifier with Training data
+ Serializing the model to disk

### Prediction on flask app
```
+ Navigate to prediction page - http://localhost:5000/predict.
+ User fill the form in the browser with required predictors value and then submit the form.
+ Values provided by users are put into model to get predictions.
+ The result from the model is showed to user on browser.
```

## To run the app
+ Open terminal in main directory '/Titanic Flask App'.
+ Then run app.py from terminal

```
python app.py
```
This will start flask app server. Now can open the site by opening http://localhost:5000<br>
Note: Right now, app runs in debug mode

## Future work
+ Improve the model
+ Provide drop down for user input
+ Take user input as string for categorical data, instead of current method i.e. encoded numeric values.
+ Improve the UI and UX.