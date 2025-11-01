import numpy as np
import pandas as pd

# file_path = 'C:/Python/PyProject/빅분기/기출 및 필수모의/고급데이터셋_pdf자료참조/' 
file_path = '/Users/yeongjunjeon/python/집중강화/'

'''
문제 1 (기본 그룹별 평균/결측치)
•           주요 컬럼: Industry, Age_Group, City, Spend
문제 설명:
1.    Industry 와 Age_Group 을 기준으로 Spend 의 평균을 구하시오.
2.    Spend 의 결측치를 해당 그룹의 평균으로 채우시오.
3.    결측치 처리 후 City 별 Spend 총합을 구하시오.

�최종 질문
Spend 총합이 가장 높은 도시(City)의 이름은?
'''

df1 = pd.read_csv(file_path + 'type1_prob01.csv')
# print(df1.info())

df1['spend_mean'] = df1.groupby(['Industry','Age_Group'])['Spend'].transform('mean')
df1['Spend'] = df1['Spend'].fillna(df1['spend_mean'])
print(df1)

spend_citysum = df1.groupby('City')['Spend'].sum()
print(spend_citysum.idxmax())



'''
문제 2 (기본 카테고리별 결측치 처리)
사용 데이터: type1_prob02.csv
❓문제
Category 별 Revenue 평균을 구하시오. 해당 평균으로 결측치를 채우시오. 연도(Year)별 Revenue 평균을 출력하시오.

�최종 질문
Revenue 평균이 가장 높은 연도는?
'''

df2 = pd.read_csv(file_path + 'type1_prob02.csv')
# print(df2)

cate_revmean = df2.groupby('Category')['Revenue'].transform('mean')
df2['Revenue'] = df2['Revenue'].fillna(cate_revmean)
yr_revmean = df2.groupby('Year')['Revenue'].mean()
print(yr_revmean.idxmax())




'''
문제 3 (transform: 그룹 내 편차)
사용 데이터: type1_prob03.csv
❓문제
Dept, Grade 별 평균 Salary 를 계산하고,
각 개인의 Salary 에서 해당 그룹의 평균을 빼서 'Diff' 컬럼을 만들고,
Diff 가 0 보다 큰 사람 수를 출력하시오

�최종 질문
평균보다 급여가 높은 사람은 몇 명인가요?

'''
df3 = pd.read_csv(file_path + 'type1_prob03.csv')

df3['gr_salary'] = df3.groupby(['Dept','Grade'])['Salary'].transform('mean')
df3['Diff'] = df3['Salary'] - df3['gr_salary']
print(len(df3[df3['Diff']>0]))
print(df3[df3['Diff']>0].shape[0]) # shape은 [0] : 행의 수, [1] : 열의 수 이다. 0,1말고는 없다.
print(len(df3[(df3['Salary'].mean() < df3['Salary'])]))






'''
문제 4 (agg: 그룹별 평균/표준편차)
사용 데이터: type1_prob04.csv
❓문제
Category, Month 별로 Sales 의 평균과 표준편차를 모두 구하시오. 평균은 'avg', 표준편차는 'std' 컬럼으로 나오게 하시오.

�최종 질문
평균 매출이 가장 높은 카테고리/월 조합은?
'''
df4 = pd.read_csv(file_path + 'type1_prob04.csv')
print(df4)

df4 = df4.groupby(['Category','Month'])['Sales'].agg(avg = 'mean', std = 'std')
print(df4)
print(df4['avg'].idxmax())



'''
문제 5 (transform: 그룹 내 표준화)
사용 데이터: type1_prob05.csv
❓문제
Product, Region 별로 Revenue 를 표준화하시오.
표준화된 값은 'Z_Score' 컬럼에 저장하시오
�최종 질문
Z_Score 의 절댓값이 가장 큰 데이터의 Product 와 Region 은?
'''

df5 = pd.read_csv(file_path + 'type1_prob05.csv')
print(df5)

gr_mean = df5.groupby(['Product','Region'])['Revenue'].transform('mean')
gr_std = df5.groupby(['Product','Region'])['Revenue'].transform('std')

df5['Z_score'] = (df5['Revenue'] - gr_mean) / gr_std

result = df5.loc[df5['Z_score'].abs().idxmax(),['Product','Region']]
print(result)

'''
문제 6 (pivot_table 활용)
사용 데이터: type1_prob06.csv
❓문제
pivot_table 을 사용하여 Month 와 Product 별 평균 Amount 를 구하시오.

�최종 질문
Amount 평균이 가장 높은 월은?
'''
df6 = pd.read_csv(file_path + 'type1_prob06.csv')
print(df6)

table = df6.pivot_table(index = 'Month', columns = 'Product', values = 'Amount', aggfunc = 'mean')
print(table)

print(table.mean(axis=1).idxmax())


'''
문제 7 (crosstab 활용)
사용 데이터: type1_prob07.csv❓문제
Age 와 Usage 별 고객 수를 crosstab 으로 구하시오. 비율을 포함한 테이블도 같이 출력하시오.

�최종 질문
가장 높은 사용 비율을 보인 연령대(Age)는?
'''

df7 = pd.read_csv(file_path + 'type1_prob07.csv')
print(df7)

customer_cnt = pd.crosstab(df7['Age'],df7['Usage'],margins=True) 
# margins = True 하면 전체 합계 All도 계산한다. False하면 All은 계산 안함.
print(customer_cnt)

customer_ratio = pd.crosstab(df7['Age'],df7['Usage'],normalize='index') 
print(customer_ratio)

print(customer_ratio.sum(axis=1).idxmax())

# print(pd.crosstab(df7['Age'], df7['Usage'], normalize='columns'))
# - index → 각 Age 그룹 내에서 Usage 비율을 계산
# - columns → 각 Usage 값 내에서 Age 비율을 계산
# - all → 전체 데이터에서 비율을 계산

'''
문제 8 (rolling 평균)
사용 데이터: type1_prob08.csv❓문제
Sales 컬럼에 대해 7 일 이동평균(rolling mean)을 구하시오. 결과는 'Rolling_Mean' 컬럼에 저장하시오.
�최종 질문
Rolling_Mean 이 가장 높은 날짜는?
'''

df8 = pd.read_csv(file_path + 'type1_prob08.csv')


df8['Rolling_Mean'] = df8['Sales'].rolling(window=7).mean()
print(df8)
print(df8.loc[df8['Rolling_Mean'].idxmax(),'Date'])


'''
문제 9 (diff 함수)
사용 데이터: type1_prob09.csv
❓문제
Price 컬럼에 대해 전일 대비 차이(Diff)를 계산하시오. 결과는 'Diff' 컬럼에 저장하시오.

�최종 질문
가장 가격이 많이 하락한 날짜는?

'''

df9 = pd.read_csv(file_path + 'type1_prob09.csv')
df9['Diff'] = df9['Price'].diff()
print(df9)
print(df9.loc[df9['Diff'].idxmin(),'Date'])



'''
문제 10 (pct_change 함수)
사용 데이터: type1_prob10.csv
❓문제
Visitors 컬럼에 대해 전주 대비 변화율(pct_change)을 계산하시오. 결과는 'Change' 컬럼에 저장하시오.

�최종 질문
가장 방문자 수 변화율이 큰 주는?
'''


df10 = pd.read_csv(file_path + 'type1_prob10.csv')
print(df10)

df10['pct_change'] = df10['Visitors'].pct_change()
print(df10.loc[df10['pct_change'].idxmax(),'Week'])