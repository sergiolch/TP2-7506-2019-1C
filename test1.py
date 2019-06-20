
#%%
import pandas as pd
import numpy as np
import datetime
import gc
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import NMF
from sklearn.externals import joblib


#%%
def fix_str_float(ds, col):
    ds[col] = ds[col].str.replace(r'[^0-9\.]','')
    ds[col] = np.where(ds[col]=='',np.nan,ds[col])
    ds[col] = ds[col].astype('float32')
    return ds[col].astype('float32')


#%%
# clicks
clicks_df = pd.read_csv('data/clicks.csv', low_memory=False,dtype={'advertiser_id':'int32','action_id':'float32','source_id':'int32','country_code':'category',                                                 'latitude':'float32','longitude':'float32','carrier_id':'float32','os_minor':'category',                                                  'os_major':'category','specs_brand':'category','timeToClick':'float32','ref_type':'category'                                                                  ,'ref_hash':'object'})

clicks_df['touchX'] = fix_str_float(clicks_df,'touchX')
clicks_df['touchY'] = fix_str_float(clicks_df,'touchY')
clicks_df['created'] = pd.to_datetime(clicks_df['created'])
#events
events_df = pd.read_csv('data/events.csv', low_memory=False, dtype={'event_id':'int32','ref_type':'category','application_id':'category',                                                                                            'attributed':'bool','device_countrycode':'category','device_city':'category',                                                                                            'trans_id':'category','carrier':'category','device_os':'category',                                                                                            'connection_type':'category'})
events_df['date'] = pd.to_datetime(events_df['date'])
events_df['wifi'].astype('bool', inplace=True)
events_df.drop(columns=['device_countrycode','session_user_agent','ip_address','device_language'], inplace=True)
# installs
installs_df = pd.read_csv('data/installs.csv', low_memory=False, dtype={'ref_type':'category','application_id':'category',                                                      'device_brand':'category','ref_hash':'object','wifi':'category'})
installs_df['kind'] = installs_df['kind'].str.lower()
installs_df['kind'] = installs_df['kind'].astype('category')
installs_df.drop(columns=['session_user_agent','ip_address','device_language','device_model'], inplace=True)
installs_df['created'] = pd.to_datetime(installs_df['created'])
installs_df.drop(['device_countrycode'], axis=1, inplace=True)
# auctions
auctions_df = pd.read_csv('data/auctions.csv', low_memory=False, dtype={'country':'category','platform':'category',                                                                        'ref_type_id':'category','source_id':'category','device_id':'object'})

auctions_df['date'] = pd.to_datetime(auctions_df['date'])
print('setup done')


#%%
installs_df.head(5)


#%%
events_df['connection_type'].describe()


#%%
events_df['connection_type'].value_counts()


#%%
events_df['connection_type'].isnull().sum()


#%%
#auctions_df = auctions_df.sort_values(by=['device_id','date'])
#auctions_df['date_dif'] = auctions_df['date'].shift(periods=-1) - auctions_df['date']
#auctions_df['in_seconds'] = np.nan
#last_row = False
#last_index = False
#for index, row in auctions_df.iterrows():
#    if not(isinstance(last_row, bool)):
#        if row['device_id']!=last_row['device_id']:
#            auctions_df.at[last_index,'date_dif'] = np.nan
#    auctions_df.at[index,'in_seconds'] = row['date_dif'].total_seconds()
#    last_row = row
#    last_index = index
#auctions_df['in_seconds'] = np.where(auctions_df['date_dif'].isnull(), np.nan, auctions_df['in_seconds'])


#%%
auctions_df.head(10)


#%%
#auctions_df.to_csv('data/auctions_seconds.csv')


#%%
#datos = pd.merge(auctions_df, installs_df, left_on='device_id', right_on='ref_hash', how='left')


#%%
installs_df.shape


#%%
auctions_df.shape


#%%
auctions_1 = auctions_df.loc[auctions_df['date']<'2019-04-20 00:00:00'].copy()


#%%
# calculate time in seconds
auctions_1.drop_duplicates(inplace=True)
auctions_1 = auctions_1.sort_values(by=['device_id','date'])
auctions_1['date_dif'] = auctions_1['date'].shift(periods=-1) - auctions_1['date']
auctions_1['device_id_next'] = auctions_1['device_id'].astype('object').shift(periods=-1)
auctions_1['date_dif'] = np.where(auctions_1['device_id_next']==auctions_1['device_id'], auctions_1['date_dif'], datetime.datetime(2019,4,20)-auctions_1['date'])
auctions_1['in_seconds'] = auctions_1['date_dif'].dt.total_seconds()
auctions_1['status_censored'] = auctions_1['device_id_next']==auctions_1['device_id']
auctions_1.drop(['device_id_next','date_dif'], axis='columns', inplace=True)
auctions_1['device_id'] = auctions_1['device_id'].astype('object')
#calculate previus time in seconds
auctions_1['date_prev'] = auctions_1['date'].shift()
auctions_1['date_dif_prev'] = auctions_1['date']- auctions_1['date_prev']
auctions_1['device_id_prev'] = auctions_1['device_id'].astype('object').shift()
auctions_1['date_dif_prev'] = np.where(auctions_1['device_id_prev']==auctions_1['device_id'], auctions_1['date_dif_prev'], auctions_1['date']-datetime.datetime(2019,4,18))
auctions_1['last_seen'] = auctions_1['date_dif_prev'].dt.total_seconds()
auctions_1.drop(['device_id_prev','date_dif_prev','date_prev'], axis='columns', inplace=True)
auctions_1 = auctions_1.sort_values(by=['date'])


#%%
auctions_1.head(5)


#%%
installs_1 = installs_df.loc[installs_df['created']<'2019-04-20 00:00:00'].copy()
installs_1.head(5)


#%%
installs_df['application_id'].describe()


#%%
installs_1.columns


#%%
# search for features
auctions_1.columns


#%%
auct_cols = auctions_1.columns.tolist()


#%%
auctions_1.shape


#%%
#information about last installs
data_1 = pd.merge(auctions_1, installs_1, left_on='device_id', right_on='ref_hash', how='left')
#only previus installs on the window
data_1 = data_1.loc[(data_1['date']>data_1['created']) | data_1['created'].isnull()]


#%%
# application_id feature by id
app_id_1 = data_1[['application_id']].copy()
app_id_1 = pd.get_dummies(app_id_1, dummy_na=True, prefix_sep='=')
data_1.drop(columns=['application_id'], inplace=True)
data_1 = pd.merge(data_1, app_id_1, left_index=True, right_index=True, how='inner')


#%%
app_id_1_columns = app_id_1.columns.tolist()


#%%
group_1 = data_1.groupby(auct_cols).agg({col:'sum' for col in app_id_1_columns})


#%%
group_1.loc[group_1['application_id=14']>2][['application_id=14']].head(5)


#%%
group_1.reset_index(inplace=True)


#%%
group_1.head(5)


#%%
##delete when no longer needed
#del events_df
#del clicks_df
#del data_1
#del app_id_1
##collect residual garbage
#gc.collect()


#%%
auctions_1 = pd.merge(auctions_1, group_1, on=['date','device_id','ref_type_id','source_id','in_seconds','status_censored', 'last_seen'], how='left')
auctions_1 = auctions_1.astype({col:'float32' for col in app_id_1_columns})


#%%
##delete when no longer needed
#del group_1
##collect residual garbage
#gc.collect()


#%%
auctions_1.fillna(value={'application_id=nan':1}, inplace=True)
auctions_1.fillna(value={col:0 for col in app_id_1_columns}, inplace=True)
auctions_1 = auctions_1.astype({col:'int32' for col in app_id_1_columns})


#%%
auctions_1.shape


#%%
#save to a file
auctions_1.to_csv('data/auctions_1.csv',index=False)


#%%
#read file
auctions_1 = pd.read_csv('data/auctions_1.csv', low_memory=False, dtype={'device_id':'object',                                                                        'ref_type_id':'category','source_id':'category','in_seconds':'float64',                                                                        'status_censored':'bool','last_seen':'float64'})
auctions_1['date'] = pd.to_datetime(auctions_1['date'])
app_cols = []
for col in auctions_1.columns:
    if col.startswith('application_id'):
        app_cols.append(col)
auctions_1 = auctions_1.astype({col:'int32' for col in app_cols})
auctions_1.dtypes


#%%
group_1.shape
#group_1.head(50)


#%%
auctions_1.columns


#%%
#data X and y
data_full_1 = pd.merge(auctions_1, installs_1, left_on='device_id', right_on='ref_hash', how='inner')
data_full_1 = data_full_1.loc[data_full_1['date']>=data_full_1['created']]
data_full_1['install_diff'] = data_full_1['date']-data_full_1['created']
data_full_1['install_seconds'] = data_full_1['install_diff'].dt.total_seconds()
data_full_1 = data_full_1.loc[data_full_1['in_seconds']>=data_full_1['install_seconds']]
data_x_1 = data_full_1.drop(columns=['in_seconds','status_censored','ref_hash'])
data_y_1 = np.fromiter((data_full_1["status_censored"], data_full_1["in_seconds"]),
                                dtype=[('status_censored', np.bool), ('in_seconds', np.float64)])


#%%
class preprocess( BaseEstimator, TransformerMixin ): 
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        X = X.copy()
        # boolean transformations
        if 'event_uuid' in X.columns:
            X['event_uuid'] = np.where(X['event_uuid'].isnull(), 0,1)
        if 'click_hash' in X.columns:
            X['click_hash'] = np.where(X['click_hash'].isnull(), 0,1)
        if 'Android' in X.columns:
            X['Android'] = np.where(X['user_agent'].str.contains('Android', regex=False),1,0)
        if 'iOS' in X.columns:
            X['iOS'] = np.where(X['user_agent'].str.contains('Darwin', regex=False) | X['user_agent'].str.contains('iOS', regex=False),1,0)
        if 'trans_id' in X.columns:
            X['trans_id'] = np.where(X['trans_id'].isnull(), 0,1)
        # date transformations
        if 'created' in X.columns:
            X['created_weekday'] = X['created'].dt.weekday
            X['created_hour'] = X['created'].dt.hour
            X['created_minute'] = X['created'].dt.minute
        if 'date' in X.columns:
            X['date_weekday'] = X['date'].dt.weekday
            X['date_hour'] = X['date'].dt.hour
            X['date_minute'] = X['date'].dt.minute
            X['date_second'] = X['date'].dt.second
        #remove unused columns
        to_drop = []
        for col in ['date','created', 'install_diff','device_brand','install_seconds','user_agent','device_id']:
            if col in X.columns:
                to_drop.append(col)
        X = X.drop(columns=to_drop)
        X = pd.get_dummies(X, dummy_na=True, prefix_sep='=')
        self.encoded_columns_ = X.columns
        #returns numpy array
        return X


#%%
# format features
#data_x_1['event_uuid'] = np.where(data_x_1['event_uuid'].isnull(), 0,1)
#data_x_1['click_hash'] = np.where(data_x_1['click_hash'].isnull(), 0,1)
#data_x_1['Android'] = np.where(data_x_1['user_agent'].str.contains('Android', regex=False),1,0)
#data_x_1['iOS'] = np.where(data_x_1['user_agent'].str.contains('Darwin', regex=False) | data_x_1['user_agent'].str.contains('iOS', regex=False),1,0)
#data_x_1['trans_id'] = np.where(data_x_1['trans_id'].isnull(), 0,1)
#data_x_1['created_weekday'] = data_x_1['created'].dt.weekday
#data_x_1['created_hour'] = data_x_1['created'].dt.hour
#data_x_1['created_minute'] = data_x_1['created'].dt.minute
#data_x_1['date_weekday'] = data_x_1['date'].dt.weekday
#data_x_1['date_hour'] = data_x_1['date'].dt.hour
#data_x_1['date_minute'] = data_x_1['date'].dt.minute
#data_x_1['date_second'] = data_x_1['date'].dt.second
#data_x_1.drop(columns=['date','created', 'install_diff','device_brand','install_seconds','user_agent'], inplace=True)
#data_x_1_numeric = pd.get_dummies(data_x_1, dummy_na=True, prefix_sep='=')


#%%
data_y_1 = np.fromiter(zip(data_full_1.head(100)["status_censored"], data_full_1.head(100)["in_seconds"]),
                                dtype=[('status_censored', np.bool), ('in_seconds', np.float64)])


#%%
estimator = CoxPHSurvivalAnalysis(alpha=0.1)
estimator.fit(data_x_1_numeric.head(100), data_y_1)
estimator.score(data_x_1_numeric.head(100), data_y_1)


#%%
def fit_and_score_features(X, y, alpha=0.1):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha=alpha)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores


#%%
scores = fit_and_score_features(data_x_1_numeric.head(100).values, data_y_1)
pd.Series(scores, index=data_x_1_numeric.columns).sort_values(ascending=False)


#%%
def custom_cv_folds(X):
    myCViterator = []
    trainIndices = X.loc[X['date']<'2019-04-18 12:00:00'].index.values.astype(int)
    testIndices =  X.loc[('2019-04-18 12:00:00'<=X['date']) & (X['date']<'2019-04-19 00:00:00')].index.values.astype(int)
    myCViterator.append( (trainIndices, testIndices) )
    trainIndices = X.loc[('2019-04-18 12:00:00'<=X['date']) & (X['date']<'2019-04-19 00:00:00')].index.values.astype(int)
    testIndices =  X.loc[('2019-04-19 00:00:00'<=X['date']) & (X['date']<'2019-04-19 12:00:00')].index.values.astype(int)
    myCViterator.append( (trainIndices, testIndices) )
    trainIndices = X.loc[('2019-04-19 00:00:00'<=X['date']) & (X['date']<'2019-04-19 12:00:00')].index.values.astype(int)
    testIndices =  X.loc[('2019-04-19 12:00:00'<=X['date']) & (X['date']<'2019-04-20 00:00:00')].index.values.astype(int)
    myCViterator.append( (trainIndices, testIndices) )
    return myCViterator


#%%
pipe = Pipeline([('preprocess', preprocess()),
                 ('reduce_dim', NMF()),
                 ('select', SelectKBest(fit_and_score_features, k=50)),
                 ('model', CoxPHSurvivalAnalysis(alpha=0.01))])


#%%
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


#%%
#auctions data
data_x_1 = auctions_1.drop(columns=['in_seconds','status_censored']).copy()
data_y_1 = np.fromiter(zip(auctions_1["status_censored"], auctions_1["in_seconds"]),
                                dtype=[('status_censored', np.bool), ('in_seconds', np.float64)])


#%%
#delete when no longer needed
#del events_df
#del clicks_df
#del data_1
#del app_id_1
#del group_1
del auctions_1
#collect residual garbage
gc.collect()


#%%
# grid search
data_x_11 = data_x_1.copy()
data_x_11.reset_index(inplace=True)
custom_cv = custom_cv_folds(data_x_11)
param_grid = {'select__k': np.arange(1, data_x_11.shape[1] + 1)}
gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=custom_cv, iid=True) #, n_jobs=-1)
gcv.fit(data_x_11, data_y_1)

pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)


#%%
gcv.best_params_


#%%
# random search
data_x_11 = data_x_1
data_x_11.reset_index(inplace=True)
custom_cv = custom_cv_folds(data_x_11)
param_grid = {'select__k': np.arange(1, data_x_11.shape[1] + 1)}
rcv = RandomizedSearchCV(pipe, param_grid, return_train_score=True, cv=custom_cv, iid=True, n_iter=10, n_jobs=2)
rcv.fit(data_x_11, data_y_1)

gc.collect() #release cache

pd.DataFrame(rcv.cv_results_).sort_values(by='mean_test_score', ascending=False)


#%%
rcv.best_params_


#%%
data_x_11 = data_x_1.head(100).copy()
pipe.set_params(**gcv.best_params_)
pipe.fit(data_x_11, data_y_1)

joblib.dump(pipe, 'data/model.sav')

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])


#%%
data_x_11 = data_x_1.head(100).copy()
pipe.score(data_x_11, data_y_1)


#%%
data_x_11.columns


#%%
data_y_1.shape


#%%
data_x_1_numeric.shape


#%%
data_y_1.shape


#%%
test1 = auctions_1[['device_id','date']]


#%%
test1 = test1.sort_values(by=['device_id','date'])


#%%
test1 = test1.head(1000)


#%%
test1.shape


#%%
test1['date_dif'] = test1['date'].shift(periods=-1) - test1['date']


#%%
test1['device_id_next'] = test1['device_id'].astype('object').shift(periods=-1)


#%%
test1['date_dif'] = np.where(test1['device_id_next']==test1['device_id'], test1['date_dif'], datetime.datetime(2019,4,20)-test1['date'])


#%%
test1['in_seconds'] = test1['date_dif'].dt.total_seconds()


#%%
test1['status-censored'] = test1['device_id_next']==test1['device_id']


#%%
test1.head(50)


#%%
test1['date_dif'] = test1['date'].shift(periods=-1) - test1['date']


#%%
test1.head(50)


#%%



