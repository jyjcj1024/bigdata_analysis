import numpy as np
import pandas as pd

'''
주어진 훈련 데이터로 머신러닝 회귀 모델을 학습하고, 테스트 데이터에 대한 매출을 예측하여 CSV로 제출합니다.
'''

df_train = pd.read_csv('/Users/yeongjunjeon/python/7회/07.02.01-sales_train_dataset.csv').copy()
df_test = pd.read_csv('/Users/yeongjunjeon/python/7회/07.02.02-sales_test_dataset_x.csv').copy()

target = df_train.pop('Sales')

print(df_test)

df_total = pd.concat([df_train,df_test]) # [] 꼭 해줘야함.
df_total = pd.get_dummies(df_total)
print(df_total)

train = df_total.iloc[:len(df_train)].copy()
test = df_total.iloc[len(df_train):].copy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_col = train.select_dtypes(include=['number']).columns

train[num_col] = scaler.fit_transform(train[num_col])
test[num_col] = scaler.transform(test[num_col])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 123)
model.fit(X_train,y_train)
pred = model.predict(X_test)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test,pred)
print(rmse)

test_predict = model.predict(test)

result = pd.DataFrame({'pred' : test_predict})
print(result)

result.to_csv('/Users/yeongjunjeon/python/7회/7회_result.csv',index=False)


from xgboost import XGBRegressor
model_xg = XGBRegressor()
model_xg.fit(X_train,y_train)

predict_xg = model.predict(test)

from scipy.stats import ttest_ind
stat,p = ttest_ind(test_predict,predict_xg)

print(stat,p)