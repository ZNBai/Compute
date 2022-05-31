#!/usr/bin/env python3

import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import os
import sys
import yaml

def read_input(input_path):
    data = pd.read_csv(input_path)
    return data

def compute_clean(input_path, output_path):
    df = read_input(input_path)
    df = df.drop(df[df.Embarked.isnull()].index)
    df.fillna(df.median(), inplace = True)
    df.to_csv(output_path + "clean.csv", index=False)
    return "Wrote the preprocessed data to: " + output_path + "clean.csv.\n" 

def compute_numeralization(input_path, output_path):
    data = read_input(input_path)
    # Factorization for class-specific attributes, Embarked,Sex,Pclass
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

    # Put new properties together
    df = pd.concat([data, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # Remove the old attributes
    df.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
    df.to_csv(output_path + "numeralization.csv", index=False)
    return "Wrote the numeralized data to: " + output_path + "numeralization.csv.\n" 

# Feature normalisation
def compute_normalization(input_path, output_path):
    df = read_input(input_path)
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
    df.to_csv(output_path + "normalization.csv", index=False)
    return "Wrote the normalized data to: " + output_path + "normalization.csv.\n" 

def compute_train(input_path, output_path):

    df = read_input(input_path)
    #df = data_clean(df)
    #df = set_numeralization(df)
    #df = set_normalization(df)

    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # Get the independent variable x
    x = train_np[:,1:]
    # Get the dependent variable y
    y = train_np[:,0]
    train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=.8)
    # Fitting using logistic regression to obtain the trained model
    clf = linear_model.LogisticRegression(solver='liblinear',C=1.0,penalty='l2',tol=1e-6)
    clf.fit(train_X,train_y)

    f = open(output_path + 'clf.pickle','wb')
    pickle.dump(clf,f)
    f.close()

    return "Accuracy score: " + str(clf.score(test_X,test_y)) + ".\nWrote the model to: " + output_path + "clf.pickle.\n" 

#def test(input: str, output_path: str) -> str:
def compute_test(input_path, model_path, output_path):
    if os.path.getsize(model_path+'clf.pickle') > 0:
        f = open(model_path+'clf.pickle','rb')
        clf = pickle.load(f)
        f.close()

        df = read_input(input_path)
        #df = data_clean(df)
        #df = set_numeralization(df)
        #df = set_normalization(df)

        test_df = df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
        predictions = clf.predict(test_df)
        result = pd.DataFrame({'PassengerId':df['PassengerId'].values,'Survived':predictions.astype(np.int32)})
        result.to_csv(output_path + "prediction.csv", index=False)

        report = "Wrote the prediction to: " + output_path + "prediction.csv.\n" 
    else:
        report = "No models exist, please train first.\n" 
    return report

# The entrypoint of the script
if __name__ == "__main__":

    # Make sure that at least one argument is given, that is either 'train' or 'test'
    if len(sys.argv) != 2 or (sys.argv[1] != "train" and sys.argv[1] != "test" and sys.argv[1] != "clean" and sys.argv[1] != "numeralization" and sys.argv[1] != "normalization"):
        print(f"Usage: {sys.argv[0]} clean|numeralization|normalization|train|test")
        exit(1)

    input_path = os.environ["INPUT"]
    output_path = os.environ["OUTPUT_PATH"]

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "clean":
        result = compute_clean(input_path, output_path)
    elif command == "numeralization":
        result = compute_numeralization(input_path, output_path)
    elif command == "normalization":
        result = compute_normalization(input_path, output_path)
    elif command == "train":
        result = compute_train(input_path, output_path)
    else:
        model_path = os.environ["MODEL_PATH"]
        result = compute_test(input_path, model_path, output_path)

        #result = test(os.environ["INPUT"], os.environ["OUTPUT_PATH"])

    # Print the result with the YAML package
    print(yaml.dump({ "output": result }))


    