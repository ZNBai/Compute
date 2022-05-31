from os import path
import sys
import Titanic

sample_train = "./t_data/train.csv"
sample_test = "./t_data/test.csv"
model_path = "./t_data/"
output_path = "./t_data/"
def test_clean():
	Titanic.clean(sample_train, output_path)
    assert path.exists(output_path + "clean.csv") is True

def test_numeralization():
	Titanic.numeralization(sample_train, output_path)
    assert path.exists(output_path + "numeralization.csv") is True

def test_normalization():
	Titanic.normalization(sample_train, output_path)
    assert path.exists(output_path + "normalization.csv") is True

def test_train():
    Titanic.train(sample_train, output_path)
    assert path.exists(output_path + "clf.pickle") is True

def test_test():
    Titanic.clean(sample_test, output_path)
    Titanic.numeralization(output_path + "clean.csv", output_path)
    Titanic.normalization(output_path + "numeralization.csv", output_path)
    Titanic.test(output_path + "normalization.csv", model_path, output_path)
    assert path.exists(output_path + "prediction.csv") is True

    
