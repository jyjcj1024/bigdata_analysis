import numpy as np
import pandas as pd


# file_path = 'C:/Python/PyProject/빅분기/기출 및 필수모의/고급데이터셋_pdf자료참조/'
file_path = '/Users/yeongjunjeon/python/집중강화/'


##### 시험점수 예측 OLS 분석

'''
�문제
다음은 학생들의 공부 시간, 수면 시간, 커피 섭취량과 시험 점수에 대한 데이터이다. 
선형회귀를 통해 시험 점수에 영향을 주는 요인을 분석하시오.
변수 설명:
•    - Study_Hours: 하루 평균 공부 시간 (시간)
•    - Sleep_Hours: 하루 평균 수면 시간 (시간)
•    - Coffee_Intake: 하루 평균 커피 섭취량 (잔)
•    - Exam_Score: 시험 점수 (종속변수)
문제:
1.    1. Exam_Score 를 종속변수로 하여 선형회귀 모델을 학습하시오.
2.    2. 각 독립변수의 회귀계수와 유의미한 변수(p-value 기준)를 해석하시오.
3.    3. R-squared 값을 해석하고, 전체 모델 설명력을 평가하시오.
'''


df1 = pd.read_csv(file_path + 'type3_study_sleep_coffee_exam.csv')
print(df1)

import statsmodels.api as sm 

X = df1[['Study_Hours','Sleep_Hours','Coffee_Intake']]
y = df1['Exam_Score']

X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())


'''
�직무성과 OLS 회귀 분석 - Full Analysis with 성과지표
�문제
다음은 직원의 경력, 학력, 근무 시간에 따른 직무 성과 점수 데이터이다. 
선형 회귀 분석을 통해 각 변수의 영향력을 평가하고, 회귀모델의 예측 성능을 분석하시오.
변수 설명:
•    - Experience: 경력 (년)
•    - Education_Level: 학력 수준 (1: 고졸, 2: 대졸, 3: 대학원졸)
•    - Hours_Per_Week: 주당 평균 근무 시간
•    - Performance_Score: 직무 성과 점수 (종속변수)
문제:
1.    1. Performance_Score 를 종속변수로 하여 선형회귀 모델을 구성하시오.
2.    2. 각 변수의 회귀계수 및 p-value 를 해석하시오.
3.    3. 결정계수(R²), MSE, RMSE, MAE 등의 회귀 모델 성능 지표를 계산하고 해석하시오
'''

df2 = pd.read_csv(file_path + 'type3_employee_performance.csv')
print(df2)

X = df2.drop(columns=['Performance_Score'])
y = df2['Performance_Score']

import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())

# 예측 성능지표
from sklearn import metrics

y_pred = model.predict(X)
R2 = metrics.r2_score(y,y_pred)
MSE = metrics.mean_squared_error(y,y_pred)
RMSE = metrics.root_mean_squared_error(y,y_pred)
MAE = metrics.mean_absolute_error(y,y_pred)

print(R2,MSE,RMSE,MAE)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())
y_pred = model.predict(X_test)

# 성능 평가
R2 = metrics.r2_score(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = MSE ** 0.5  # RMSE 직접 계산
MAE = metrics.mean_absolute_error(y_test, y_pred)
print(R2, MSE, RMSE, MAE)


'''
로지스틱 회귀 분석 – 흡연과 건강
문제 설명
흡연(Smoking), 운동 빈도(Exercise), 음주 빈도(Alcohol) 세 가지 생활습관 변수를 사용하여 
전반적 건강 상태(General_Health: 0=양호, 1=불량)를 예측하는 로지스틱 회귀 모델을 구축하고, 
혼동행렬, 분류 지표, ROC-AUC 로 성능을 평가한다.
'''

df3 = pd.read_csv(file_path + 'type3_health_logit.csv')
print(df3)

y = df3['General_Health']
X = df3.drop(columns='General_Health')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.Logit(y_train,X_train).fit()

# scikit-learn은 X, y
# statsmodels는 y, X

# 라이브러리	모델 종류	함수	인자 순서
# scikit-learn	RandomForest, SVM 등	.fit(X, y)	입력(X), 타겟(y)
# statsmodels	Logit, OLS 등	Logit(y, X)	타겟(y), 입력(X)

print(model.summary())

pred_prob = model.predict(X_test)
pred = (pred_prob>0.5).astype(int)
print(pred)

from sklearn import metrics

confm = metrics.confusion_matrix(y_test,pred)
report = metrics.classification_report(y_test,pred)
auc = metrics.roc_auc_score(y_test,pred_prob)

print(confm)
print(report)
print(auc)

