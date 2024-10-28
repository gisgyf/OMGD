import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import omgd

# 设置随机种子以确保结果可复现
np.random.seed(0)

# 生成正态分布数据
n = 50
data_normal = np.random.normal(0, 1, n)

# 定义相关性列表
correlations = [0.8, 0.6, 0.4, 0.2]

# 创建一个空的DataFrame来存储所有数据
df = pd.DataFrame({'normal': data_normal})

# 为每个相关性生成数据
for corr in correlations:
    # 生成独立的正态分布数据
    independent_data = np.random.normal(0, 1, n)

    # 使用公式生成相关数据
    correlated_data = corr * data_normal + np.sqrt(1 - corr ** 2) * independent_data

    # 将数据添加到DataFrame
    df[f'corr{corr}'] = correlated_data

# 打印DataFrame的前几行
print(df.head())

# 验证相关性
print("\n相关性矩阵:")
print(df.corr())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

omgd_result = omgd.omgd(df, Y=df.columns[0], factors=df.columns[1:], n_variates=1, disc_interval=[3])
print(omgd_result['factor'])

df.to_csv(f'data/simulation{n}.csv', index=False)