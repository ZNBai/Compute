from os import path
import sys
import Titanic

sample_train = "tests/data/train.csv"
sample_test = "tests/data/test.csv"
model_path = "tests/data/"
output_path = "tests/data/"

def test_train():
    Titanic.train(sample_train, model_path)
    assert path.exists(model_path + "clf.pickle") is True

def test_test():
    Titanic.test(sample_test, model_path, output_path)
    assert path.exists(output_path + "prediction.csv") is True
    
