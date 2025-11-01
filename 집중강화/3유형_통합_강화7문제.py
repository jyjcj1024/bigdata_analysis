import numpy as np
import pandas as pd


# file_path = 'C:/Python/PyProject/빅분기/기출 및 필수모의/고급데이터셋_pdf자료참조/'
file_path = '/Users/yeongjunjeon/python/집중강화/3유형 데이터셋/'

#문제1

df1 = pd.read_csv(file_path + 'mock3_type_problem1.csv')
print(df1)

X = df1[['age','study_hours']]
y = df1['grade']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model = sm.OLS(y_train,X_train_const).fit()
print(model.summary())
#----- 유의하지 않은 개수 없다

df_spr = df1[['study_hours','grade']]
corr = df_spr.corr()
print(corr['grade']['study_hours'])

pred = model.predict(X_test_const)

from sklearn import metrics
rmse = metrics.root_mean_squared_error(y_test,pred)
print(rmse)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
pred_ln = model.predict(X_test)
print(metrics.root_mean_squared_error(pred,pred_ln))
print(pred)
print(pred_ln)




#문제2
df2 = pd.read_csv(file_path + 'mock3_type_problem2.csv')
print(df2)

from scipy import stats
Male = df2[df2['gender']=='M']['income']
Female = df2[df2['gender']=='F']['income']
stat,p = stats.ttest_ind(Male,Female,equal_var = True)
print(stat,p)


l_stat,l_p = stats.levene(Male,Female)
print(l_stat,l_p)

Male_norm = stats.shapiro(Male)
Female_norm = stats.shapiro(Female)
print(Male_norm.statistic, Male_norm.pvalue)
print(Female_norm.statistic, Female_norm.pvalue)

#문제3
df3 = pd.read_csv(file_path + 'mock3_type_problem3.csv')
print(df3)

y = df3['y']
X = df3[['x1','x2']]

import statsmodels.api as sm
X_const = sm.add_constant(X)
model = sm.OLS(y,X_const).fit()
print(model.summary())



from scipy import stats
resid = model.resid
resid_stat,resid_p = stats.shapiro(resid)
print(resid_stat,resid_p)

from statsmodels.stats.diagnostic import het_breuschpagan
print(het_breuschpagan(resid,X_const))





# import statsmodels
# import pprint
# pprint.pprint(dir(statsmodels.stats.diagnostic))

#문제4
df4 = pd.read_csv(file_path + 'mock3_type_problem4.csv')
print(df4)

treat = df4[df4['treatment'] == 1]['recovery_days']
ntreat = df4[df4['treatment'] == 0]['recovery_days']

from scipy.stats  import ttest_ind,shapiro
stat,p = ttest_ind(treat,ntreat,equal_var = True)
print(stat,p)

a = df4[(df4['treatment'] == 1) & (df4['recovered']==1)].shape[0]
b = df4[(df4['treatment'] == 0) & (df4['recovered']==1)].shape[0]
c = df4[(df4['treatment'] == 1) & (df4['recovered']==0)].shape[0]
d = df4[(df4['treatment'] == 0) & (df4['recovered']==0)].shape[0]

# 각 그룹의 총 개수 계산
treated_total = df4[df4['treatment'] == 1].shape[0]  # 처치군 전체 개수
control_total = df4[df4['treatment'] == 0].shape[0]  # 대조군 전체 개수

# 사건 발생 확률 계산
treated_recovery_prob = a / treated_total  # 처치군에서 회복한 확률
control_recovery_prob = b / control_total  # 대조군에서 회복한 확률

treated_not_recovered_prob = c / treated_total  # 처치군에서 회복하지 못한 확률
control_not_recovered_prob = d / control_total  # 대조군에서 회복하지 못한 확률

# 오즈비 계산
odds_ratio = (treated_recovery_prob / treated_not_recovered_prob) / (control_recovery_prob / control_not_recovered_prob)
## 결국 (a/b) / (c/d) 이거다.
print(odds_ratio)

treat_normp = shapiro(treat).pvalue
ntreat_normp = shapiro(ntreat).pvalue
print(treat_normp, ntreat_normp)


#문제5
df5 = pd.read_csv(file_path + 'mock3_type_problem5.csv')
print(df5)

df5['certified_dum'] = df5['certified'].map({'Y' : 1, 'N' : 0})
print(df5)

import statsmodels.api as sm

y = df5['salary']
X = df5[['experience','certified_dum']]
X_const = sm.add_constant(X)

model = sm.OLS(y,X_const).fit()
print(model.pvalues)
print(model.params)


corr = df5[['experience','salary']].corr()
print(corr['salary']['experience'])

certi_Y = df5[df5['certified']=='Y']['salary']
certi_N = df5[df5['certified']=='N']['salary']

from scipy.stats import ttest_ind

stat,p = ttest_ind(certi_N,certi_Y)
print(stat,p)




#문제6
df6 = pd.read_csv(file_path + 'mock3_type_problem6.csv')
print(df6)

print(df6.corr()['satisfaction']['usage'])


from scipy.stats import shapiro

stat,p = shapiro(df6['usage'])
print(stat,p)

Q1 = df6['usage'].quantile(0.25)
Q3 = df6['usage'].quantile(0.75)
IQR = Q3-Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
normal_df6 = df6[(lower <= df6['usage']) & (df6['usage'] <= upper)]

gap = df6['usage'].mean() - normal_df6['usage'].mean()
print(gap)




#문제7
df7 = pd.read_csv(file_path + 'mock3_type_problem7.csv')
print(df7)

dummy = pd.get_dummies(df7['contract_type'],prefix = 'contract', drop_first=True)
total = pd.concat([df7,dummy],axis=1).drop(columns='contract_type',axis=1).astype(int)
print(total)

y = total['churn']
X = total.drop(columns='churn',axis=1)
print(X.dtypes, y.dtypes)

import statsmodels.api as sm

X_const = sm.add_constant(X)
model = sm.Logit(y,X_const).fit()
print(model.summary())
print(round(model.params['support_calls'],3)) # 출력 0.7497이 반올림되서 0.750
print(f"{round(0.7497, 3):.3f}")  # 출력: 0.750
print(f'contract odds ratio : {np.exp(model.params['contract_one-year'])}')

pred = model.predict(X_const)
print((pred>0.5).sum())


from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# 훈련 데이터와 테스트 데이터 분리 (예: 80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 모델 훈련
X_train_const = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train_const).fit(disp=False)

# 테스트 데이터로 예측
X_test_const = sm.add_constant(X_test)
pred = model.predict(X_test_const)

# 예측 결과 분석
print((pred > 0.5).sum())  # 테스트 데이터에서 해지 예측된 고객 수 출력




#문제7
df7 = pd.read_csv(file_path + 'mock3_type_problem7.csv')
print(df7)

import statsmodels.formula.api as smf

model = smf.logit('churn ~ age + support_calls + monthly_usage + C(contract_type)', data=df7).fit()
print(model.summary())
print(np.exp(model.params['support_calls']))





# #문제3
# df3 = pd.read_csv(file_path + 'mock3_type_problem3.csv')
# print(df3)

# y = df3['y']
# X = df3[['x1','x2']]

# import statsmodels.api as sm
# X_const = sm.add_constant(X)
# model = sm.OLS(y,X_const).fit()
# print(model.summary())

# import statsmodels.formula.api as smf
# model =smf.ols('y~x1+x2',df3).fit()
# print(model.summary())
# print(model.resid)

