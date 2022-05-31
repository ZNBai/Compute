from os import path
import sys
import Titanic

sample_train = "./t_data/train.csv"
sample_test = "./t_data/test.csv"
model_path = "./t_data/"
output_path = "./t_data/"
def test_clean():
	Titanic.clean(sample_train, output_path)
    assert path.exists(model_path + "clean.csv") is True

def test_numeralization():
	Titanic.numeralization(sample_train, output_path)
    assert path.exists(model_path + "numeralization.csv") is True

def test_normalization():
	Titanic.normalization(sample_train, output_path)
    assert path.exists(model_path + "normalization.csv") is True

def test_test():
    Titanic.test(sample_test, model_path, output_path)
    assert path.exists(output_path + "prediction.csv") is True

def test_train():
    Titanic.train(sample_train, model_path)
    assert path.exists(model_path + "clf.pickle") is True

    
