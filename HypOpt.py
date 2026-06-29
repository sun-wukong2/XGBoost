import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns

from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import matplotlib.pyplot as plt

train = pd.read_csv(r'C:\Users\PHou\PycharmProjects\XGBoost\bike.csv')
train['datetime'] = pd.to_datetime( train['datetime'] )
train['day'] = train['datetime'].map(lambda x: x.day)

##Modeling

def assing_test_samples(data, last_training_day=0.3, seed=1):
    days = data.day.unique()
    np.random.seed(seed)
    np.random.shuffle(days)
    test_days = days[: int(len(days) * 0.3)]

    data['is_test'] = data.day.isin(test_days)


def select_features(data):
    columns = data.columns[(data.dtypes == np.int64) | (data.dtypes == np.float64) | (data.dtypes == bool)].values
    return [feat for feat in columns if feat not in ['count', 'casual', 'registered'] and 'log' not in feat]


def get_X_y(data, target_variable):
    features = select_features(data)

    X = data[features].values
    y = data[target_variable].values

    return X, y


def train_test_split(train, target_variable):
    df_train = train[train.is_test == False]
    df_test = train[train.is_test == True]

    X_train, y_train = get_X_y(df_train, target_variable)
    X_test, y_test = get_X_y(df_test, target_variable)

    return X_train, X_test, y_train, y_test


def fit_and_predict(train, model, target_variable):
    X_train, X_test, y_train, y_test = train_test_split(train, target_variable)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return (y_test, y_pred)


def post_pred(y_pred):
    y_pred[y_pred < 0] = 0
    return y_pred


def rmsle(y_true, y_pred, y_pred_only_positive=True):
    if y_pred_only_positive: y_pred = post_pred(y_pred)

    diff = np.log(y_pred + 1) - np.log(y_true + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


assing_test_samples(train)


def etl_datetime(df):
    df['year'] = df['datetime'].map(lambda x: x.year)
    df['month'] = df['datetime'].map(lambda x: x.month)

    df['hour'] = df['datetime'].map(lambda x: x.hour)
    df['minute'] = df['datetime'].map(lambda x: x.minute)
    df['dayofweek'] = df['datetime'].map(lambda x: x.dayofweek)
    df['weekend'] = df['datetime'].map(lambda x: x.dayofweek in [5, 6])


etl_datetime(train)

train['{0}_log'.format('count')] = train['count'].map(lambda x: np.log2(x))

for name in ['registered', 'casual']:
    train['{0}_log'.format(name)] = train[name].map(lambda x: np.log2(x + 1))


##Hyperparameter Optimization

def objective(space):
    model = xgb.XGBRegressor(
        max_depth=int(space['max_depth']),
        n_estimators=int(space['n_estimators']),
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        learning_rate=space['learning_rate'],
        reg_alpha=space['reg_alpha']
    )

    X_train, X_test, y_train, y_test = train_test_split(train, 'count')
    eval_set = [(X_train, y_train), (X_test, y_test)]

    (_, registered_pred) = fit_and_predict(train, model, 'registered_log')
    (_, casual_pred) = fit_and_predict(train, model, 'casual_log')

    y_test = train[train.is_test == True]['count']
    y_pred = (np.exp2(registered_pred) - 1) + (np.exp2(casual_pred) - 1)

    score = rmsle(y_test, y_pred)
    print
    "SCORE:", score

    return {'loss': score, 'status': STATUS_OK}


space = {
    'max_depth': hp.quniform("x_max_depth", 2, 20, 1),
    'n_estimators': hp.quniform("n_estimators", 100, 1000, 1),
    'subsample': hp.uniform('x_subsample', 0.8, 1),
    'colsample_bytree': hp.uniform('x_colsample_bytree', 0.1, 1),
    'learning_rate': hp.uniform('x_learning_rate', 0.01, 0.1),
    'reg_alpha': hp.uniform('x_reg_alpha', 0.1, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials)

print(best)
