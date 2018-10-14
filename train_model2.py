## Importing

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from xgboost import XGBClassifier

sns.set(rc={"figure.figsize": (12, 8)})

## N


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Secelcts used columns
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s"
                           % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

df_train = pd.read_csv('train.csv', index_col='PassengerId')
df_test = pd.read_csv('test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], keys=["train", "test"])


df.columns = list(df.columns.str.lower())

df.cabin = df.cabin.str[0]
df['title'] = df['name'].apply(lambda c: c[c.index(',')+2:c.index('.')])
df.title[df.title.isin(['Miss', 'Mlle'])] = 'Miss'
df.title[df.title.isin(['Mrs', 'Ms', 'Mme'])] = 'Ms'
df.title[~df.title.isin(['Miss', 'Ms', 'Mr', 'Master'])] = 'Other'
df.cabin = df.cabin.fillna('Z')

category_features = ['cabin', 'embarked', 'title', 'pclass', 'sex']

X = (
    df
    .drop('survived', axis=1)
    .astype('float', errors='ignore')
)

for f in category_features:
    X[f] = X[f].astype('category').cat.codes

y = df.survived

X_train = X.loc['train']
X_test = X.loc['test']

y_train = y.dropna()


x_cols = ['age', 'cabin', 'embarked', 'fare', 'parch', 'pclass',
          'sex', 'sibsp', 'title']


preprocess_pipeline = make_pipeline(
    ColumnSelector(columns=x_cols),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector('float64'), # np.number
            Imputer(strategy="median"),
            StandardScaler()
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            Imputer(strategy="most_frequent"),
            OneHotEncoder()
        ))
    ])
)
classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    SVC(kernel="rbf", random_state=42)
)

param_grid = {
    "svc__gamma": [0.1 * x for x in range(1, 6)]
}

classifier_model = GridSearchCV(classifier_pipeline, param_grid, cv=10)
classifier_model.fit(X_train, y_train)
