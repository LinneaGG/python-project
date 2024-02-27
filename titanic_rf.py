# Random forest model to predict survival on the Titanic 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn import set_config
import matplotlib.pyplot as plt

# Read data into pandas DataFrame object 
titanic_data = pd.read_csv(
    "C:/Users/Linnea Good/Documents/PhD/Pythonkurs/project/train.csv")

# The data I want to predict after I have built the model 
titanic_test_data = pd.read_csv(
    "C:/Users/Linnea Good/Documents/PhD/Pythonkurs/project/test.csv")

# Set PassengerId as index in DataFrame 
titanic_data = titanic_data.set_index("PassengerId")

titanic_test_data = titanic_test_data.set_index("PassengerId")

# Remove columns that will not help classify (probably) 
titanic_data = titanic_data.drop("Name", axis="columns")
titanic_data = titanic_data.drop("Ticket", axis="columns")
titanic_data = titanic_data.drop("Cabin", axis="columns")

titanic_test_data = titanic_test_data.drop("Name", axis="columns")
titanic_test_data = titanic_test_data.drop("Ticket", axis="columns")
titanic_test_data = titanic_test_data.drop("Cabin", axis="columns")

# Make Female = 1 and Male = 0 
titanic_data['Sex'] = titanic_data['Sex'].map({'male':0,'female':1})
titanic_data = titanic_data.rename(columns={"Sex": "Female"})

titanic_test_data['Sex'] = titanic_test_data['Sex'].map({'male':0,'female':1})
titanic_test_data = titanic_test_data.rename(columns={"Sex": "Female"})

# Create dummy variables for Embarked 
titanic_data = pd.get_dummies(titanic_data)

titanic_test_data = pd.get_dummies(titanic_test_data)

# Split the data into features (X) and target (Y)
X = titanic_data.drop("Survived", axis=1)
Y = titanic_data["Survived"] 

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Impute age to handle NaNs 
print("Proportion of NaNs in Age column: " + str(titanic_data["Age"].isna().sum() / len(titanic_data["Age"])))

set_config(transform_output="pandas") # To keep imputed output as pandas DF

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)
titanic_test_data = imp.transform(titanic_test_data)

rf = RandomForestClassifier()

# Hyperparameter tuning 
hyperparameters = {'n_estimators': randint(200,800),
              'max_depth': randint(20,40)}

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = hyperparameters, 
                                 n_iter=10, 
                                 cv=5)

rand_search.fit(X_train_imp, Y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
Y_pred = best_rf.predict(X_test_imp)

# Create the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Create bar plot of most important features in the model 
feature_importances = pd.Series(
    best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
plt.figure() # But first make a new plot window 
feature_importances.plot.bar();

# And finally: Use the trained model to predict the test data 
# Need to first merge with DataFrame to keep index 
titanic_test_data["Survived"] = best_rf.predict(titanic_test_data) 
Survival_pred = titanic_test_data[["Survived"]]

# Save predictions to submit to Kaggle 
Survival_pred.to_csv("Survival.csv", index=True)