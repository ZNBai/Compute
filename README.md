# Brane Scikit-learn package
![example event parameter](https://github.com/ZNBai/Compute/blob/main/.github/workflows/test.yml/badge.svg?event=push) [![DOI](https://zenodo.org/badge/497735628.svg)](https://zenodo.org/badge/latestdoi/497735628)
## Requirements
pyyaml==5.4.1
pandas==1.2.4
numpy==1.20.3
sklearn==1.1.1
flake8
## Installation
```
brane import ZNBai/Compute
```
## Finctions
| NAME | INPUT | OUTPUT |
| :----: | :----: | :----: |
|  clean   |  INPUT: '/data/train.csv'<br>OUTPUT_PATH: '/data/' | /data/clean.csv |
| numeralization  |  INPUT: '/data/train.csv'<br>OUTPUT_PATH: '/data/' | /data/numeralization.csv |
| normalization  |  INPUT: '/data/train.csv'<br>OUTPUT_PATH: '/data/' | /data/normalization.csv |
| train  |  INPUT: '/data/train.csv'<br>OUTPUT_PATH: '/data/' | clf.pickle |
| test  |  INPUT: '/data/train.csv'<br>MODEL_PATH: '/data/'<br>OUTPUT_PATH: '/data/' | /data/prediction.csv |
## Tests
Run unit tests with pytest: `pytest`
