
#Common imports

from __future__ import division, print_function, unicode_literals

import os

import numpy as np

import pickle

np.random.seed(42)

import pandas as pd

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

 

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

 

PROJECT_ROOT_DIR = "."

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Images")

DOC_PATH = os.path.join(PROJECT_ROOT_DIR, "Data_and_models")

 

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)

 

 

 

Preprocessed_2019 = pd.read_csv(DOC_PATH + '/ProjectDetailsOncorCleanedFULL_Preprocessed.csv')

Preprocessed_2019 = Preprocessed_2019.drop(Preprocessed_2019.columns[[0]], axis=1)

df = Preprocessed_2019

timeline_target = df

 

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')

df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

 

catfull=[

]

numfull=['’]

date=['Status Date']

leg='’

 

cat = catfull

num = numfull

stringvar=cat

floatvar=num

 

#Changing dtype

for col in [stringvar]:

    df[col] = df[col].astype('str')

for col in [floatvar]:

    df[col] = df[col].astype('float64')  

    

#Dataframe selector

from sklearn.impute import SimpleImputer

from sklearn import preprocessing

 

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self,X):

        return X[self.attribute_names].values

 

 

#Forecast Future Trend - SARIMAX

 

import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'

 

timeline_target = timeline_target.dropna(subset=['Days_New_to_InterconnectionApproved'])

target = timeline_target

for col in ['Interconnection Approved_statusdate']:

    target[col] = target[col].astype('datetime64')

    

target = target.sort_values('Interconnection Approved_statusdate')

 

#target['Interconnection Approved_statusdate'].min(), target['Interconnection Approved_statusdate'].max()

 

target = target[['Interconnection Approved_statusdate','Days_New_to_InterconnectionApproved']]

target = target.groupby('Interconnection Approved_statusdate')['Days_New_to_InterconnectionApproved'].count().reset_index()

target = target.set_index('Interconnection Approved_statusdate')

 

y = target['Days_New_to_InterconnectionApproved'].resample('MS').sum()

y['2019':]

 

 

y.plot(figsize=(15, 6))

plt.show()

 

 

from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

#save_fig('Components_2019_DG_Growth')

plt.show()

 

 

p = d = q = range(0, 2)

 

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]

#seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

 

 

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

print('\n')

 

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)

           

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue

 

 

# #weekly and weekly_mean()

# mod = sm.tsa.statespace.SARIMAX(y,

#                                 order=(1, 1, 1),

#                                 seasonal_order=(1, 1, 0, 52),

#                                 enforce_stationarity=False,

#                                 enforce_invertibility=False)

 

 

 

#Monthly

mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

 

# #Monthly_mean()

# mod = sm.tsa.statespace.SARIMAX(y,

#                                 order=(1, 1, 0),

#                                 seasonal_order=(1, 1, 0, 12),

#                                 enforce_stationarity=False,

#                                 enforce_invertibility=False)

 

 

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))

plt.show()

 

 

pred = results.get_prediction(start=pd.to_datetime('2019'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2018-06':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Number of IA-approved Projects')

plt.legend()

plt.show()

 

 

y_forecasted = pred.predicted_mean

y_truth = y['2019':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

 

 

pred_uc = results.get_forecast(steps=6)

pred_ci = pred_uc.conf_int()

ax = y['2019':].plot(label='observed',figsize=(14, 7))

#ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('ONCOR')

ax.set_ylabel('Number of IA-approved Projects')

plt.legend()

#save_fig('ONCOR_Approved_Projects_Growth_Trend')

plt.show()