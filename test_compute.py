from os import path
import sys
import compute

sample_train = "./t_data/train.csv"
sample_test = "./t_data/test.csv"
model_path = "./t_data/"
output_path = "./t_data/"

def test_clean():
    compute.compute_clean(sample_train, output_path)
    assert path.exists(output_path + "clean.csv") is True

def test_numeralization():
    compute.compute_numeralization(sample_train, output_path)
    assert path.exists(output_path + "numeralization.csv") is True

def test_normalization():
    compute.compute_normalization(sample_train, output_path)
    assert path.exists(output_path + "normalization.csv") is True

def test_train():
    compute.compute_train(sample_train, output_path)
    assert path.exists(output_path + "clf.pickle") is True

def test_test():
    compute.compute_test(sample_test, model_path, output_path)
    assert path.exists(output_path + "prediction.csv") is True

    
