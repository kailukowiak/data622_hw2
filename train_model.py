## Loading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.externals import joblib

# Reading Files in
df_train = pd.read_csv('train.csv', index_col='PassengerId')
df_test = pd.read_csv('test.csv', index_col='PassengerId')
df = pd.concat([df_train, df_test], keys=["train", "test"], sort=True)
# Merging the two datasets gives us an easy way to make sure all our transform
# Are done correctly. Subsetting them in this way also makes it easy to split
# out.

#  Cleaning Names
df.Cabin = df.Cabin.str[0]  # Because the rest is mostly useless.
df['Title'] = df['Name'].apply(lambda c: c[c.index(',')+2:c.index('.')])
# Titles give some extra info but there are redundancies that have little
# Explanitory value
df.Title[df.Title.isin(['Miss', 'Mlle'])] = 'Miss'
df.Title[df.Title.isin(['Mrs', 'Ms', 'Mme'])] = 'Ms'
df.Title[~df.Title.isin(['Miss', 'Ms', 'Mr', 'Master'])] = 'Other'
df.Cabin = df.Cabin.fillna('Z')  # Z because we can treat all na the same
df = df.drop(['Name', 'Ticket'], axis=1)  # further, we don't want either of
# the above values because they are almost random/unique.

# The following code creates categorical variables:
def dummier(df, dummies_list):
    df[dummies_list] = df[dummies_list].astype('category', order=True)


dummies_list = ['Cabin', 'Embarked', 'Title', 'Pclass',
                'Sex', 'Survived']

cat_levels = df[dummies_list].astype('category')

cat_dict = {}
for i in cat_levels.columns:
    cat_dict[i] = list(cat_levels[i].cat.categories)

# Imputations based on medians for numeric stuff
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

# Setting categories to dummy variables because sklearn handles these better

dummy_cols = df.select_dtypes(include='object').columns
dummies = pd.get_dummies(df[dummy_cols], drop_first=True)  # drop first to avoid
# multi coliniarity

df.drop(dummy_cols, axis=1, inplace=True)  # drop them as we already have dummy
df = pd.concat([df, dummies], axis=1)  # Merge together

# Split into train and test
train = df.loc['train']
test = df.loc['test']
test = test.drop('Survived', axis=1)
test.to_pickle('test.pkl')  # save as pickle file
# The above could have been done through CSV but pandas is more reliable. 

# Training matrix (numpy)
X = train.drop('Survived', axis=1).values
y = train['Survived'].values

# Random Forrest
rfc = RandomForestClassifier(random_state=42)
# The param grid is ideal for seaching for the best hyper parameters
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X, y)
best_rf = CV_rfc.best_estimator_  
# After a grid search we chose the best model (based on cross validation)
# and then dump it into a joblib file.
joblib.dump(best_rf, 'clf.joblib')
# I choose to save these as a joblib because it handels models better than
# A pickl file would
