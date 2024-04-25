from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


train_data = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Titanic-Disaster/data/train.csv')
# print(train_data.head())
test_data = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Titanic-Disaster/data/test.csv')
# print(test_data.head())

# the meaning of the columns is as follows:
# Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
# so we first use loc to get the data of female, and then get the Survived column, and then calculate the rate of survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print(rate_women)

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
# filter Nan
train_data = train_data[features].dropna()
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Titanic-Disaster/data/my_submission.csv', index=False)
