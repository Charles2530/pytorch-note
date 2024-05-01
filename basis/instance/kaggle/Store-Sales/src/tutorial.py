from statsmodels.tsa.deterministic import CalendarFourier
from warnings import simplefilter
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
import numpy as np
from pathlib import Path

simplefilter(action='ignore')
comp_dir = Path(__file__).resolve().parents[1].joinpath('data')
# print(comp_dir)
train_df = pd.read_csv(comp_dir.joinpath('train.csv'),
                       usecols=['store_nbr', 'family',
                                'date', 'sales', 'onpromotion'],
                       dtype={
                           'store_nbr': 'category',
    'family': 'category',
    'sales': np.float32,
    'onpromotion': np.uint32
},
    parse_dates=['date'],  # parse_date is used to convert date to datetime
    infer_datetime_format=True,  # to speed up datetime parsing
)
# below used 'D' to convert date to datetime,'D' means day
train_df['date'] = train_df.date.dt.to_period('D')
# print(train_df.head())
train_df = (train_df.set_index(
    ['date', 'family', 'store_nbr']
).sort_index())
# print(train_df.head())
test_df = pd.read_csv(comp_dir.joinpath('test.csv'),
                      dtype={
    'store_nbr': 'category',
    'family': 'category',
    'onpromotion': np.uint32
},
    parse_dates=['date'],
    infer_datetime_format=True
)
test_df['date'] = test_df.date.dt.to_period('D')
test_df = (test_df.set_index(
    ['date', 'family', 'store_nbr']
).sort_index())
holidays_events = pd.read_csv(comp_dir.joinpath('holidays_events.csv'),
                              dtype={
    'type': 'category',
    'locate': 'category',
    'locale_name': 'category',
    'description': 'category',
    'transferred': 'bool',
},
    parse_dates=['date'],
    infer_datetime_format=True
)
holidays_events = holidays_events.set_index('date').to_period('D')
# print(holidays_events.head())
"""Baseline Submission"""
X_train = train_df.copy()
y_train = (
    X_train.unstack(['family', 'store_nbr']).loc[:, "sales"]
)
index_ = X_train.index.get_level_values('date').unique()
STORE = '1'
FAMILY = 'BREAD/BAKERY'
START = '2016-01-01'
END = '2016-06-15'
y_true = (y_train
          .stack(['family', 'store_nbr'])
          .to_frame()
          .query('family == @FAMILY and store_nbr == @STORE')
          .reset_index(['family', 'store_nbr'], drop=True)
          .rename(columns={0: 'ground_truth'})
          .loc[START:END, :]
          .squeeze()
          )

dp = DeterministicProcess(
    index=index_,
    constant=True,
    order=1,
    drop=True
)
X_time = dp.in_sample()
X_time_test = dp.out_of_sample(steps=16)
X_time_test.index.name = 'date'
base_model = joblib.load(Path(__file__).resolve(
).parents[1].joinpath('model', 'baseline_model.pkl'))
y_submit = pd.DataFrame(
    base_model.predict(X_time_test),
    index=X_time_test.index,
    columns=y_train.columns
).clip(0.0)
baseline_submission = (y_submit.stack(['family', 'store_nbr']).
                       to_frame().join(test_df.id)
                       .rename(columns={0: 'sales'})
                       .reset_index(drop=True)
                       .reindex(columns=['id', 'sales'])
                       )
baseline_submission.to_csv(
    comp_dir.joinpath('baseline_submission.csv'), index=False)

"""submission with seasons accounted"""
X = y_true.to_frame()
X["day"] = X.index.dayofweek
X["week"] = X.index.week
X_seasonal = dp.in_sample()
day_of_week = pd.Series(X_seasonal.index.dayofweek, index=index_)
X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
X_seasonal = pd.concat([X_seasonal, X_day_of_week], axis=1)
fourier = CalendarFourier(freq='A', order=10)
dp_fourier = DeterministicProcess(
    index=index_,
    constant=False,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)
X_fourier = dp_fourier.in_sample()
X_seasonal = pd.concat([X_seasonal, X_fourier], axis=1)

holidays = (holidays_events.query('transferred == False')
            .query("locale=='National'")
            .loc[:, 'description']
            .to_frame()
            .assign(description=lambda x: x.description.cat.remove_unused_categories())
            )
duplicated_dates = holidays.index.duplicated(keep='first')
holidays = holidays[~duplicated_dates]
X_holidays = pd.get_dummies(holidays)
X_seasonal = X_seasonal.join(X_holidays, on='date', how='left').fillna(0.0)

X_seasonal_test = dp.out_of_sample(steps=16)
X_seasonal_test.index.name = 'date'
day_of_week = pd.Series(X_seasonal_test.index.dayofweek,
                        index=X_seasonal_test.index)
X_day_of_week = pd.get_dummies(day_of_week, prefix='day_of_week')
X_seasonal_test = pd.concat([X_seasonal_test, X_day_of_week], axis=1)

X_fourier_test = dp_fourier.out_of_sample(steps=16)
X_seasonal_test = pd.concat([X_seasonal_test, X_fourier_test], axis=1)

X_seasonal_test.index.name = 'date'
X_seasonal_test = X_seasonal_test.join(
    X_holidays, on='date', how='left').fillna(0.0)

season_model = joblib.load(Path(__file__).resolve(
).parents[1].joinpath('model', 'seasonal_model.pkl'))
y_seasonal_forecast = pd.DataFrame(
    season_model.predict(X_seasonal_test),
    index=X_seasonal_test.index,
    columns=y_train.columns
).clip(0.0)
seasonal_submission = (y_seasonal_forecast.stack(['family', 'store_nbr'])
                       .to_frame().join(test_df.id)
                       .rename(columns={0: 'sales'})
                       .reset_index(drop=True)
                       .reindex(columns=['id', 'sales'])
                       )
seasonal_submission.to_csv(
    comp_dir.joinpath('seasonal_submission.csv'), index=False)
"""submission with vulcano"""


def make_lags(ts, lags, prefix=None):
    return pd.concat({
        f'{prefix}_lag_{lag}': ts.shift(lag) for lag in lags
    }, axis=1)


vulcano = pd.DataFrame(
    (X_time.index == '2016-04-16')*1.0,
    index=index_,
    columns=['vulcano'])
X_vulcano_ = make_lags(vulcano.squeeze(), lags=range(22), prefix='vulcano')
X_vulcano_ = X_vulcano_.fillna(0.0)
X_vulcano = pd.concat([X_seasonal, X_vulcano_], axis=1)
X_vulcano_test = X_seasonal_test.join(X_vulcano_, how='left').fillna(0.0)
vulcano_model = joblib.load(Path(__file__).resolve(
).parents[1].joinpath('model', 'vulcano_model.pkl'))
y_vulcano_forecast = pd.DataFrame(
    vulcano_model.predict(X_vulcano_test),
    index=X_vulcano_test.index,
    columns=y_train.columns
).clip(0.0)
vulcano_submission = (y_vulcano_forecast.stack(['family', 'store_nbr'])
                      .to_frame().join(test_df.id)
                      .rename(columns={0: 'sales'})
                      .reset_index(drop=True)
                      .reindex(columns=['id', 'sales'])
                      )
vulcano_submission.to_csv(
    comp_dir.joinpath('vulcano_submission.csv'), index=False)


# """submission with multistep target"""


def make_multistep_target(ts, steps):
    return pd.concat({
        f'y_step_{step+1}': ts.shift(-step) for step in range(steps)
    }, axis=1)


def fetch_forecast(data, START='2017-08-16', END='2017-08-31'):
    X = data.loc[START]
    DATES = pd.period_range(START, END)
    index_to_rename = X.index.get_level_values(0).unique()[:len(DATES)]
    rename_dict = dict(zip(index_to_rename, DATES))
    forecast = X.rename(rename_dict, level=0).to_frame()
    forecast.index = forecast.index.set_names('date', level=0)
    return forecast


y_train_multi = make_multistep_target(y_train, steps=16).dropna()
y_train_multi, X_vulcano_cut = y_train_multi.align(
    X_vulcano, join='inner', axis=0)
model_multi = joblib.load(Path(__file__).resolve(
).parents[1].joinpath('model', 'model_multi.pkl'))
y_multi_fit = pd.DataFrame(
    model_multi.predict(X_vulcano_cut),
    index=X_vulcano_cut.index,
    columns=y_train_multi.columns
).clip(0.0)
y_multi_forecast = pd.DataFrame(
    model_multi.predict(X_vulcano_test),
    index=X_vulcano_test.index,
    columns=y_train_multi.columns
).clip(0.0)
multi_submission = (
    fetch_forecast(y_multi_forecast).join(test_df.id).reset_index(drop=True)
)
multi_submission['sales'] = multi_submission.iloc[:, 0]
multi_submission = multi_submission[['id', 'sales']]
multi_submission.to_csv(
    comp_dir.joinpath('multi_submission.csv'), index=False)
