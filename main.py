import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

#### Import DATA #############
df = pd.read_csv('train.csv')
he = pd.read_csv('holidays_events.csv')
it = pd.read_csv('items.csv')
st = pd.read_csv('stores.csv')

############ Creating a subset ########## 
df = df[df['date']>'2015-08-01']
st = st.rename(columns={'type': 'store_type'})

######## Merge data frames ################
df = df.merge(st, how='inner', on='store_nbr')
df = df.merge(he[he['locale']=='Local'], how='left', left_on = ['date','city'], right_on = ['date','locale_name'])
df = df.merge(he[he['locale']=='Regional'], how='left', left_on = ['date','state'], right_on = ['date','locale_name'])
df = df.merge(he[he['locale']=='National'], how='left', left_on = ['date'], right_on = ['date'])
df = df.merge(it, how='inner', left_on = ['item_nbr'], right_on = ['item_nbr'])
df = df.merge(oil, how='inner', left_on = ['date'], right_on = ['date'])

######### Cleanup merge ###############
df['holiday'].unique() = df[df['type_x'].notnull()]['type_x'].apply(str)+df[df['type_y'].notnull()]['type_y'].apply(str)+df[df['type'].notnull()]['type'].apply(str)
df['holiday'] = df['holiday'].apply(lambda x: x.replace('nan',''))
df['transfer'] = df['transferred_x'].apply(str)+df['transferred_y'].apply(str)+df['transferred'].apply(str)
df['transfer'] = df['transfer'].apply(lambda x: x.replace('nan',''))

################ Drop unnecessray columns ######### 
df = df.set_index('date')
df.index =  pd.to_datetime(df.index)
df['day'] = df.index.day
df['dayofyear'] = df.index.dayofyear
df = df.drop(['family','state','city','locale','locale_x','locale_y',
              'locale_name','locale_name_x','locale_name_y',
              'description','description_x','description_y',
              'type_x','type_y','type','transferred_x','transferred_y','transferred'],axis=1)
df = df.fillna('')


########## Label encode string values #############
le1 = preprocessing.LabelEncoder()
df['store_type'] = le1.fit_transform(df['store_type'])   
le2 = preprocessing.LabelEncoder()
df['transfer'] = le2.fit_transform(df['transfer']) 
le3 = preprocessing.LabelEncoder()
df['holiday'] = le3.fit_transform(df['holiday'])   
le4 = preprocessing.LabelEncoder()
df['onpromotion'] = le4.fit_transform(df['onpromotion'])   
 

############ Split Train and Test datasets ################       
dfstheit_train = df[df.index<='2017-05-01']
dfstheit_train = dfstheit_train.set_index('id')
dfstheit_train_y = dfstheit_train['unit_sales']
dfstheit_train = dfstheit_train.drop('unit_sales',axis=1)

dfstheit_test = df[df.index>'2017-05-01']
dfstheit_test = dfstheit_test.set_index('id')
dfstheit_test_y = dfstheit_test['unit_sales']
dfstheit_test = dfstheit_test.drop('unit_sales',axis=1)


########### Train a GBM Model ################
model = GradientBoostingRegressor(loss="huber", n_estimators=200)
clf = model.fit(dfstheit_train.values,dfstheit_train_y.values)


############ Check accuracy ################ 
pred = clf.predict(dfstheit_test)#[:,1]
mse = mean_squared_error(dfstheit_test_y, pred)
rmse = np.sqrt(mse)
print(rmse)
