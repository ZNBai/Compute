from os import path
import sys
from Titanic import titanic

sample_train = "tests/data/train.csv"
sample_test = "tests/data/test.csv"
output_path = "tests/data/"

def train_titanic():
    titanic.train(sample_train)
    assert path.exists("tests/data/clf.pickle") is True

def test_titanic():
    titanic.test(sample_test, output_path)
    assert path.exists("tests/data/prediction.csv") is True
    
