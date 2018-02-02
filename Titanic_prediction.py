# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:59:33 2017

@author: manisha
"""
# Import the Pandas library
import pandas as pd
# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
#--------------------------------------------------------------------
# Load the train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ------------------------------------------------------------------
# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

#----------------------------------------------------------------------------------
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])
# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

#-------------------------------------------------------------------------------------
#Cleaning and Formatting Data...
# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')
test["Embarked"] = test["Embarked"].fillna('S')
# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])
#-----------------------------------------------------------------------------------
#Decision Tree one

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score(mean accuracy) of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

#------------------------------------------------------------------
#Prediction one
# Impute the missing value with the median
test.Fare[152] = test.Fare.median()
print(test.Fare)
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction_one = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_one = pd.DataFrame(my_prediction_one, PassengerId, columns = ["Survived"])
print(my_solution_one)

#----------------------------------------------------------------------------
#Decision Tree two
# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))
# Impute the missing value with the median
test.Fare[152] = test.Fare.median()
print(test.Fare)
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features_two = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

# Make your prediction using the test set
my_prediction = my_tree_two.predict(test_features_two)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_two = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution_two)


#----------------------------------------------------------------------------
#Decision Tree three
# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train["Parch"] + train["SibSp"] + 1
test_two = test.copy()
test_two["family_size"] = test["Parch"] + test["SibSp"] + 1
# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_three = my_tree_three.fit(features_three, target)
#Print the score of the new decison tree
print(my_tree_three.score(features_three, target))
# Impute the missing value with the median
test_two.Fare[152] = test.Fare.median()
print(test_two.Fare)
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features_three = test_two[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "family_size"]].values

# Make your prediction using the test set
my_prediction = my_tree_two.predict(test_features_three)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_three = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution_three)


#----------------------------------------------------------------------------------
#Random Forest
# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train_two[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
# Look at the importance and score(mean accuracy) of the included features
print(my_forest.feature_importances_)
print(my_forest.score(features_forest, target))


# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test_two[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values

# Make your prediction using the test set
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)  

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
