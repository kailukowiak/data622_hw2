import subprocess
import pandas as pd
from sklearn.externals import joblib


test_df = pd.read_pickle('test.pkl')
test = test_df.values

clf = joblib.load('clf.joblib')

est = clf.predict(test)

submission = pd.DataFrame({"PassengerId": test_df.index,
                           "Survived": est})
submission['Survived'] = submission.Survived.astype('int64')

submission.to_csv('submission.csv', index=False)


command = 'kaggle competitions submit titanic -f submission.csv -m "Test Submission1"'
# subprocess.Popen(command)

subprocess.check_output(command, shell=True)
#  subprocess.check_call(command.split())

results_command = 'kaggle competitions submissions titanic'
# results = subprocess.check_call(results_command.split())
results = subprocess.check_output(results_command.split())
results = results.decode("utf-8")

with open('results.txt', 'w') as text_file:
    text_file.write(results)
