## Loading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.externals import joblib


## Loading data
df_train = pd.read_csv('train.csv', index_col='PassengerId')
df_test = pd.read_csv('test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], keys=["train", "test"], sort=True)

## Cleaning Names
df.Cabin = df.Cabin.str[0]
df['Title'] = df['Name'].apply(lambda c: c[c.index(',')+2:c.index('.')])
df.Title[df.Title.isin(['Miss', 'Mlle'])] = 'Miss'
df.Title[df.Title.isin(['Mrs', 'Ms', 'Mme'])] = 'Ms'
df.Title[~df.Title.isin(['Miss', 'Ms', 'Mr', 'Master'])] = 'Other'
df.Cabin = df.Cabin.fillna('Z')
df = df.drop(['Name', 'Ticket'], axis=1)
## Break

def dummier(df, dummies_list):
    df[dummies_list] = df[dummies_list].astype('category', order=True)


dummies_list = ['Cabin', 'Embarked', 'Title', 'Pclass',
                'Sex', 'Survived']

cat_levels = df[dummies_list].astype('category')

cat_dict = {}
for i in cat_levels.columns:
    cat_dict[i] = list(cat_levels[i].cat.categories)

## Done


def imputer(df, groupby_list):
    def impute_median(series):
        return series.fillna(series.median())

    for col_name in df.columns:
        if col_name in groupby_list:
            pass
        elif df[col_name].dtype in ['float64', 'int64']:
            df[col_name] = df.groupby(groupby_list)[col_name]\
                             .transform(impute_median)
        else:
            pass

    return df


df = imputer(df, ['Pclass', 'Sex'])
df.Embarked[df.Embarked.isna()] = 'C'  # mostt common for the class they were.
df.info()

## Classes

dummy_cols = df.select_dtypes(include='object').columns
dummies = pd.get_dummies(df[dummy_cols], drop_first=True)

df.drop(dummy_cols, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
print(df.head())
## End

train = df.loc['train']
test = df.loc['test']
test.to_pickle('test.pkl')

## TEsting

X = train.drop('Survived', axis=1).values
y = train['Survived'].values

# rf = RandomForestClassifier(n_estimators=100)
# scores = cross_val_score(rf, X, y, cv=10, scoring="accuracy")



## Training

rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X, y)
best_rf = CV_rfc.best_estimator_

joblib.dump(best_rf, 'clf.joblib')
