from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train_data = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/train.csv')
print(train_data.head())
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP']
train_X = train_data[features].fillna(0)
train_y = train_data['Transported']

train_X = pd.get_dummies(train_X)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X, train_y)

test_data = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/test.csv')
test_X = test_data[features].fillna(0)
test_X = pd.get_dummies(test_X)
predictions = model.predict(test_X)

output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Transported': predictions})
output.to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Spaceship-Titanic/data/random_forest_submission.csv', index=False)
