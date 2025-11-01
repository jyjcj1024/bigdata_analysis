import numpy as np
import pandas as pd

'''
� 문제
어떤 종목의 과거 일별 주가 데이터를 바탕으로 다음날 종가가 상승할지 하락할지를 예측하는 분류 모델을 만드시오.
- 주어진 데이터는 Open, High, Low, Close, Volume 컬럼으로 구성되어 있음
- Target 컬럼은 다음날 종가가 오늘보다 상승했는지 여부를 1(상승), 0(하락)으로 표현 - 분류 모델을 훈련하여 Target 을 예측하고,
  실제값과 비교하여 정확도(accuracy), 혼동행렬, AUC 를 평가하시오.
� 데이터 예시 컬럼
Date, Open, High, Low, Close, Volume, Target

'''


file_path = 'C:/Python/PyProject/빅분기/기출 및 필수모의/고급데이터셋_pdf자료참조/'
df = pd.read_csv(file_path + 'type2_stock_prediction.csv')
print(df)

target = df.pop('Target')
df.drop('Date',axis=1,inplace=True)

train = df

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=123)
model.fit(X_train,y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

from sklearn import metrics

report = metrics.classification_report(y_test,pred)
print(report)

accuracy = metrics.accuracy_score(y_test,pred)
print(accuracy)

f1_score = metrics.f1_score(y_test,pred,average='macro')
print(f1_score)

confusion = metrics.confusion_matrix(y_test,pred)
print(confusion)

auc = metrics.roc_auc_score(y_test,prob)
print(auc)

'''
좋은 질문이야! model.predict_proba(X_test)를 그대로 사용하면 모든 클래스에 대한 확률값을 포함한 배열이 반환돼.
하지만 [:,1]을 추가하면 특정 클래스(양성 클래스, 즉 1)에 대한 확률만 추출하는 거야.
1. model.predict_proba(X_test)의 구조
이 함수는 각 샘플에 대해 모든 클래스(0과 1)의 확률값을 반환해.
예를 들어 이진 분류에서는 다음과 같은 구조가 나와:
array([[0.3, 0.7],  # 첫 번째 샘플 → 클래스 0의 확률=0.3, 클래스 1의 확률=0.7  
       [0.6, 0.4],  # 두 번째 샘플 → 클래스 0의 확률=0.6, 클래스 1의 확률=0.4  
       [0.2, 0.8]]) # 세 번째 샘플 → 클래스 0의 확률=0.2, 클래스 1의 확률=0.8  


2. [:,1]을 추가하는 이유
AUC 계산에서는 양성 클래스(1)의 확률을 기준으로 ROC 곡선을 만들기 때문이야.
따라서 [:,1]을 사용해서 클래스 1의 확률만 따로 추출해야 해.
prob = model.predict_proba(X_test)[:,1]


이렇게 하면 다음과 같은 값이 저장돼:
array([0.7, 0.4, 0.8])


즉, 각 샘플이 1일 확률만 저장되므로 roc_auc_score() 함수에서 올바른 평가가 가능해!
3. model.predict_proba(X_test) 그대로 사용하면 안 되나?
❌ 안 돼!
그냥 사용하면 모든 클래스(0과 1)의 확률이 포함된 2D 배열이므로 roc_auc_score(y_test, prob)에서 오류가 나거나 잘못된 값이 나올 수 있어.
정리
- model.predict_proba(X_test) → 각 클래스(0과 1)의 확률을 포함 (2D 배열)
- model.predict_proba(X_test)[:,1] → 클래스 1의 확률만 추출 (AUC 계산에 필요)


- ROC 곡선은 다양한 임계값에서 모델의 **참 양성률(TPR)**과 **거짓 양성률(FPR)**을 비교하는 그래프야.
- 임계값을 바꾸면서 모델이 ‘양성(1)’이라고 판단할 기준을 높였다 낮췄다 하는 거지.
- 이 과정에서 필요한 데이터는 양성 클래스(1)에 대한 확률값이야.
→ predict_proba()[:,1]을 사용해 클래스 1의 확률을 가져오는 이유가 바로 이거야!

predict_proba(X_test) =
[[0.2, 0.8],  # 첫 번째 샘플 → 클래스 1일 확률 0.8
 [0.6, 0.4],  # 두 번째 샘플 → 클래스 1일 확률 0.4
 [0.3, 0.7]]  # 세 번째 샘플 → 클래스 1일 확률 0.7

→ 여기서 [:,1]을 하면 [0.8, 0.4, 0.7]만 추출돼!
이 확률값을 기반으로 임계값을 다양하게 설정해 ROC 곡선을 그리고, 그 아래 면적을 **AUC(Area Under the Curve)**로 계산하는 거야.
정리
🔹 AUC는 ROC 곡선 아래 면적을 의미
🔹 ROC 곡선을 만들려면 ‘양성 클래스(1)’의 확률값이 필요
🔹 그래서 predict_proba()[:,1]을 사용해야 함!
이제 좀 더 명확해졌을까? 😊 추가 질문 있으면 언제든 물어봐!

'''