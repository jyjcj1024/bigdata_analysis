import numpy as np
import pandas as pd
from scipy import stats

#descirbe
data = [10,20,30,40,50]
print(stats.describe(data))

#ttest_1samp

sample = [48,49,50,51,52]
result = stats.ttest_1samp(sample,popmean=60,alternative='greater')
print(result.pvalue)

result = stats.ttest_1samp(sample,popmean=60,alternative='two-sided')
print(result.pvalue)

result = stats.ttest_1samp(sample,popmean=60,alternative='less')
print(result.pvalue)



# ✅ 3. ttest_ind
group1 = [60, 62, 64, 66, 68]
group2 = [70, 72, 74, 76, 78]
# 🎯 H₀: 두 집단의 평균은 같다 / H₁: group2가 같지않다.
print("3️⃣ levene (등분산검정):", stats.levene(group1, group2))
print("3️⃣ ttest_ind:", stats.ttest_ind(group1, group2, equal_var=True, alternative='two-sided'))
print("3️⃣ ttest_ind:", stats.ttest_ind(group1, group2, equal_var=True, alternative='greater'))
print("3️⃣ ttest_ind:", stats.ttest_ind(group1, group2, equal_var=True, alternative='less'))

print("3️⃣ ttest_ind:", stats.ttest_ind(group1, group2)) #아무것도 없으면 등분산이라 가정하고 대립가설 양측으로 한다.



# ✅ 4. ttest_rel
before = [100, 102.3, 98.2, 101.9, 99.3]
after = [103.23, 104, 101.3, 86.4, 114]

print("4️⃣ ttest_rel:", stats.ttest_rel(before, after, alternative='greater'))
print("4️⃣ ttest_rel:", stats.ttest_rel(before, after, alternative='less'))
print("4️⃣ ttest_rel:", stats.ttest_rel(before, after, alternative='two-sided'))

# 📌 해석 정리
# t값 = -0.298 → 큰 차이 아님
# p값 > 0.05 전부 넘음 → 통계적으로 유의미한 차이 없다
# 즉, before와 after는 평균 차이가 있다고 보기 어려워 😥


# ✅ 7. f_oneway
g1 = [50, 55, 60]
g2 = [65, 70, 75]
g3 = [80, 85, 90]
# 🎯 H₀: 모든 그룹 평균이 같다 / H₁: 적어도 한 그룹은 다르다
print("7️⃣ f_oneway (ANOVA):", stats.f_oneway(g1, g2, g3))

#pvalue = 0.001, 즉 3개 중 평균이 다른 그룹이 1개 이상 있다.

# ✅ f_oneway만 하면?

# H₀: 모든 그룹 평균이 같다
# H₁: 적어도 한 그룹은 다르다
# → 결과가 **유의미(p < 0.05)**하면
# 👉 "어느 그룹이 다른가?"는 모르지! 그래서 사후검정 필요

# 먼저 crosstab해줘야함

# 1. 값과 그룹 라벨을 붙이자!
values = g1 + g2 + g3
groups = ['g1'] * len(g1) + ['g2'] * len(g2) + ['g3'] * len(g3)

df = pd.DataFrame({'score': values, 'group': groups})
print(df)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
# 2. 사후검정 (Tukey HSD)
result = pairwise_tukeyhsd(df['score'], df['group'], alpha=0.05)
print(result)

#(모두 reject=True면 전부 평균이 유의하게 다르다는 뜻이야!) 
# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#     g1     g2     15.0 0.0242  2.4738 27.5262   True
#     g1     g3     30.0 0.0008 17.4738 42.5262   True
#     g2     g3     15.0 0.0242  2.4738 27.5262   True
# ----------------------------------------------------


# ✅ 11. chi2_contingency

# 한 마트에서 고객들의 성별과 구매 여부 간의 관련성을 분석하고자 한다.
# 아래는 100명의 고객을 대상으로 조사한 교차표이다.
# 구매함	구매 안 함
# 남자	20	30
# 여자	35	15
# 🎯 이 때, "성별과 구매 여부가 독립적인가?"를 카이제곱 검정으로 판단하자.
table = np.array([[20, 30], [35, 15]])
# 🎯 H₀: 두 범주형 변수는 독립이다 / H₁: 독립이 아니다
chi2, p, dof, expected = stats.chi2_contingency(table) 
# 그룹화된 데이터 교차표를 넣어도 됨. cross = pd.crosstab(df['성별'], df['구매여부'])
# chi2, p, dof, expected = chi2_contingency(cross)

print("chi2_contingency:", (chi2, p, dof))
print("기대도수:\n", expected)

# 🎯 가설 정리

# H₀ (귀무가설): 성별과 구매 여부는 독립이다
# H₁ (대립가설): 성별과 구매 여부는 관련이 있다
# 🧪 결과 해석

# p-value = 0.00489 < 0.05
# → 유의수준 5%에서 귀무가설 기각!
# ✅ 결론
# 성별과 구매 여부는 독립이 아니다.
# 즉, 성별에 따라 구매 행동이 달라진다는 통계적으로 유의미한 증거가 있어! 🎉


# ✅ 12. pearsonr
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# 🎯 H₀: 상관 없음 / H₁: 상관 있다
print("1️⃣2️⃣ pearsonr:", stats.pearsonr(x, y))


# ✅ 13. spearmanr
x2 = [1, 3, 2, 4, 5]
y2 = [5, 3, 4, 2, 1]
# 🎯 H₀: 순위 상관 없음 / H₁: 상관 있음
print("1️⃣3️⃣ spearmanr:", stats.spearmanr(x2, y2))


# ✅ 15. zscore
matrix = np.array([[1, 2, 3], [4, 5, 6]])
# 📌 평균 0, 표준편차 1로 정규화된 z-score
print("1️⃣5️⃣ zscore:\n", stats.zscore(matrix, axis=0))
