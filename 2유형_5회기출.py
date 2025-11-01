import numpy as np
import pandas as pd

# train_set = pd.read_csv('C:/Python/PyProject/24년프로젝트/12.빅분기실기/빅분기/data/유형모음/기출문제5회_2유형_train.csv')
# test_set = pd.read_csv('C:/Python/PyProject/24년프로젝트/12.빅분기실기/빅분기/data/유형모음/기출문제5회_2유형_test.csv')

df_train = pd.read_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/기출문제5회_2유형_train.csv').copy()
df_test = pd.read_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/기출문제5회_2유형_test.csv').copy()


'''
다음에 주어진 데이터셋은 중고차와 관련된 데이터이다.  
학습용 데이터를 이용하여 중고차의 판매 가격을 예측하고, 평가용 데이터에 대한 예측 결과를 csv파일로 제출하시오. (모델의 성능은 RMSE로 평가)
'''

#target 떼어내기
target = df_train.pop('price')



# df_train = train_set.copy()
# df_test = test_set.copy()

# ID 제거
df_train.drop('ID', axis=1, inplace=True)
df_test.drop('ID', axis=1, inplace=True)



#원핫인코딩 및 겟더미
df_total = pd.concat([df_train,df_test])
df_total = pd.get_dummies(df_total)

print(df_total)

train = df_total.iloc[:len(df_train)].copy()
test = df_total.iloc[len(df_train):].copy()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#수치형 열 
col_num = train.select_dtypes(include=['number']).columns

train[col_num] = scaler.fit_transform(train[col_num])
test[col_num] = scaler.transform(test[col_num])


#모델링
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size = 0.2)


from xgboost import XGBRegressor
### 주의!!! classifier가 아니라 Regressor를 했기 때문에, 수치적인 연속적인 데이터다. 이런 경우는 분류레포트가 나올 수가 없다.
## 회귀 검증은 mean_absolute_error나 mean_square_error를 사용해야한다.

model = XGBRegressor()

model.fit(X_train, y_train)

pred = model.predict(X_test)

from sklearn import metrics

report = metrics.root_mean_squared_error(y_test,pred)
print(report)

#최종 예측아웃풋
pred_result = model.predict(test)

result = pd.DataFrame({'pred' : pred_result})

print(result)

result.to_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/5회_result.csv',index=False)


# 번외로 랜덤포레스트 회귀를 한다면

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=123)
# model = RandomForestRegressor(n_estimators=100, max_depth=4)
# model = RandomForestRegressor(random_state=123)
model.fit(X_train,y_train)

pred = model.predict(X_test)

report = metrics.root_mean_squared_error(y_test,pred)

print(report)


pred_result = model.predict(test)

result = pd.DataFrame({'pred':pred_result})

result.to_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/5회_result(random_foreset_regressor).csv',index=False)
