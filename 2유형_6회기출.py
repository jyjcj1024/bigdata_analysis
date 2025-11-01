'''
2유형 : 무조건 다 맞춘다는 생각으로 외울 것은 외워버릴 것
6회 기출

'''
import numpy as np
import pandas as pd

# df_train = pd.read_csv('C:/Python/PyProject/24년프로젝트/12.빅분기실기/빅분기/data/유형모음/기출문제6회_2유형_train.csv')
# df_test = pd.read_csv('C:/Python/PyProject/24년프로젝트/12.빅분기실기/빅분기/data/유형모음/기출문제6회_2유형_test.csv')
df_train = pd.read_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/기출문제6회_2유형_train.csv').copy()
df_test = pd.read_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/기출문제6회_2유형_test.csv')

print(df_train)
print(df_test)


# 학습용 데이터 타겟 값 떼어내기 : 이거하면 df_train에는 더이상 'General_Health가 없다.
train_target = df_train.pop('General_Health')
print(train_target)




# ID같은 건 삭제
df_train.drop('ID',axis=1,inplace=True)
df_test.drop('ID',axis=1,inplace=True)



###### 구분  ########	설명
# ❓ 원핫인코딩	범주형 데이터를 숫자 벡터로 변환하는 방법
# 🧠 왜 필요?	모델이 문자를 이해 못 하니까 숫자로 바꿔줘야 해
# 💥 왜 concat?	train/test에 있는 범주가 다를 수 있어서! 같이 인코딩해야 열 구조가 같아짐
# 🧰 pd.get_dummies()	판다스에서 자동으로 원핫인코딩해주는 함수야 💕


# 💡 원핫인코딩은 "값"을 기준으로 "컬럼"을 새로 만드는 거야
# 예를 들어 Checkup이라는 원래 컬럼엔 이런 값들이 있을 수 있어:
# bash
# 복사
# 편집
# 'Within the past year'
# '5 or more years ago'
# 'Within the past 2 years'
# 'Never'
# 'Within the past 5 years'
# 👉 이걸 pd.get_dummies() 하면:

# Checkup_Within the past year	Checkup_5 or more years ago	Checkup_Never	...

# 이렇게 각 값마다 하나의 새로운 열(column) 이 생겨!
# 그래서 Checkup 1개가 여러 열로 쪼개지는 거야!

# 🔍 그러니까 왜 train/test에는 없던 컬럼이 생기냐면:
# df_train에는 "Checkup_Never"가 없었는데
# df_test엔 "Checkup_Never"가 있는 경우처럼,

# 합쳐서 처리하면 전체 범주가 다 드러나서 열이 더 많아지는 거야!

# 그래서 get_dummies() 하고 나면:

# plaintext
# 복사
# 편집
# 원래 컬럼 18개 👉 더미 변수 포함해서 40~50개로 늘어남!
# 🌈 한눈에 보기
# 구분	컬럼 개수	예시
# 🔹 df_train.columns	18개	'Checkup', 'Sex', 'Age_Category' 등 원본 그대로
# 🔹 df_test.columns	18개	마찬가지
# 🔹 df_total_encoded.columns	50개 이상	Checkup_~, Sex_Male, Sex_Female, Age_Category_18-24 등등 분해된 버전

# ## 행은 train과 test를 합친 갯수다.
# print(len(df_train))
# print(len(df_test))
# print(len(df_total))

df_total = pd.concat([df_train,df_test])
df_total = pd.get_dummies(df_total)

# print(df_train.columns)
# print(df_test.columns)
# print(df_total.columns)


## train와 test 영역 나눠주기 반드시 필요!!!!!!!!!!!!!!
train = df_total.iloc[:len(df_train)].copy()
test = df_total.iloc[len(df_train):].copy()


## min_max 스켈링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#수치형만 할 수 있다.
num_col = train.select_dtypes(include=['number']).columns

train[num_col] = scaler.fit_transform(train[num_col]) # train이 학습용이니까 train기준으로만 fit_transform 해서 스켈링한다.
test[num_col] = scaler.transform(test[num_col]) # 학습용으로 스켈링 된 모델 기준으로 test데이터도 스켈링 적용하는거다.


### 자 이제 머신러닝

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, train_target, test_size=0.2) # train과 test 8:2로 한다는 소리

# 입력 데이터
# train: 독립 변수(features)로, 모델에 입력되는 데이터.
# train_target: 종속 변수(target)로, 모델이 예측하려는 값.
# 출력 데이터
# X_train: train 데이터의 80%를 학습용으로 분리.
# X_test: train 데이터의 20%를 테스트용으로 분리.
# y_train: train_target의 80%를 학습용으로 분리.
# y_test: train_target의 20%를 테스트용으로 분리.

# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=123)

# 모델 피팅
model.fit(X_train, y_train)
pred = model.predict(X_test)

#평가
from sklearn import metrics

report = metrics.classification_report(y_test, pred) # ★순서 중요!! '정답을 예측과 비교'한다. 라고 이해하자. 순서 기억!
print(report)

f1_score = metrics.f1_score(y_test,pred,average='macro')
print(f1_score)


# 이 모델로 실제 예측해보자.
pred_test = model.predict(test)

result = pd.DataFrame({'pred' : pred_test})
print(result)

# index를 제거하고 넣으려면 반드시 index=False를 해줘야한다.
# result.to_csv('C:/Python/PyProject/24년프로젝트/12.빅분기실기/빅분기/data/유형모음/예측결과.csv',index=False)
result.to_csv('/Users/yeongjunjeon/python/빅분기/data/유형모음/6회_예측결과.csv',index=False)


