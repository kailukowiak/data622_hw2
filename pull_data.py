# This action requires kaggle to be downloaded
# Kaggle is imported as a check to better throw an error
import subprocess
import pandas as pd
try:
    import kaggle
except ModuleNotFoundError:
    subprocess.check_call('pip install kaggle'.split())

command = 'kaggle competitions download -c titanic'
subprocess.check_call(command.split())


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## This tests for the size and correct names:

train_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
               'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

assert list(train.columns) == train_names, "different column names"
assert train.shape == (891, 12), 'Train has wrong shape'

## test for test names

test_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
              'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

assert list(test.columns) == test_names, 'Wrong Column Names'
assert test.shape == (418, 11), 'Train has wrong shape'
