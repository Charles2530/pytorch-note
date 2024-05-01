import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
df = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Store-Sales/data/train.csv')
# print(df.head())
# print(df.columns)


"""clean data"""

# transform date to datetime to extract week and year
df['date'] = pd.to_datetime(df['date'])
# isocalendar() returns a tuple of year, week and day
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['weekname'] = df['date'].dt.day_name()
df['promo-sales'] = df['onpromotion']/df['sales']
# print(df.loc[0:5, ['date', 'week', 'year', 'weekname', 'promo-sales']])

"""organize data"""

target = 'sales'
category = [x for x in df.columns if df[x].dtype == 'object']
num = [x for x in df.columns if df[x].dtype in ('int64', 'float64')]
num.remove('sales')
num.remove('id')
# print(num)

"""split data"""
train_df = df.loc[df.year != 2017]
test_df = df.loc[df.year == 2017]
# print(train_df.shape, test_df.shape)

"""Data Science"""
# print(train_df.groupby('family')[target].describe().sort_values(by='std'))


class DataSelector(BaseEstimator, TransformerMixin):
    """Select columns from a DataFrame"""

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


num_pp = Pipeline([
    ('data_selector', DataSelector(['onpromotion'])),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

category_pp = Pipeline([
    ('data_selector', DataSelector(['family', 'store_nbr'])),
    ('encoder', OneHotEncoder(sparse_output=False))
])

pipe = FeatureUnion([
    ('num', num_pp),
    ('category', category_pp)
])

pipe.fit(train_df)
train_pp = pipe.transform(train_df)
test_pp = pipe.transform(test_df)

"""Modeling"""
reg_lin = LinearRegression()
reg_lin.fit(train_pp, train_df[target])
reg_svr = LinearSVR()
reg_svr.fit(train_pp, train_df[target])
print("Baseline Linear Regression: ", np.sqrt(-cross_val_score(
    reg_lin, train_pp, train_df[target], scoring='neg_mean_squared_error').mean()))
print("Baseline SVR: ", np.sqrt(-cross_val_score(
    reg_svr, train_pp, train_df[target], scoring='neg_mean_squared_error').mean()))

sample = pd.read_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Store-Sales/data/test.csv')
pd.DataFrame({
    'id': sample['id'],
    'sales': reg_lin.predict(pipe.transform(sample))
}).to_csv(
    '/home/charles/charles/python/pytorch/project/basis/instance/kaggle/Store-Sales/data/submission.csv', index=False)
