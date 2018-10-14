import pandas as pd
from sklearn.externals import joblib

test = pd.read_pickle('test.pkl')
test = test.values

clf = joblib.load('clf.joblib')
