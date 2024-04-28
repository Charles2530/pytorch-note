from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Path of the file to read
train = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Housing-Prices/data/train.csv')
print(train.shape)
# print(train.head())
target = train.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF',
            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = train[features].dropna(axis=1)
train_X = pd.get_dummies(train_X)
# print(train_X.head())

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X, target)
test = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Housing-Prices/data/test.csv')
test_X = test[features].dropna(axis=1)
test_X = pd.get_dummies(test_X)
# print(test_X.head())
predictions = model.predict(test_X)
output = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
output.to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Housing-Prices/data/my_submission.csv', index=False)
print("Your submission was successfully saved!")
