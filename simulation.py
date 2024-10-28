import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import omgd

# set random seed for reproducibility
np.random.seed(0)

# define a function to generate and standardize data
def generate_data(size, distribution):
    if distribution == 'normal':
        data = np.random.normal(loc=0, scale=1, size=size)
    elif distribution == 't':
        data = np.random.standard_t(df=30, size=size)
    elif distribution == 'uniform':
        data = np.random.uniform(low=0.0, high=1.0, size=size)
    elif distribution == 'gamma':
        data = np.random.gamma(2, 2, size)

    # 标准化数据
    return (data - data.mean()) / data.std() * 10

# generate data
sample_sizes = [300, 500, 700]
distributions = ['normal', 't', 'uniform', 'gamma']
data_sets = {size: {dist: generate_data(size, dist) for dist in distributions}
             for size in sample_sizes}

# create graph and subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

for i, size in enumerate(sample_sizes):
    for j, (dist, title) in enumerate(zip(distributions,
                                          ['Normal', 'T', 'Uniform', 'Gamma'])):
        axes[i, j].hist(data_sets[size][dist], bins=20, edgecolor='black')
        axes[i, j].set_title(f'{title} (n={size})')
        axes[i, j].set_xlabel('Value')
        axes[i, j].set_ylabel('Frequency')


# print statistical information
for size in sample_sizes:
    print(f"\nSample size: {size}")
    for dist, name in zip(distributions, ['Normal', 'T', 'Uniform', 'Gamma']):
        data = data_sets[size][dist]
        print(f"{name} distribution:")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Standard deviation: {data.std():.4f}")
        print(f"  Skewness: {stats.skew(data):.4f}")

# Create sorted DataFrames
dataframes = {}
for size in sample_sizes:
    df_data = {}
    for dist in distributions:
        df_data[dist] = np.sort(data_sets[size][dist])
    dataframes[size] = pd.DataFrame(df_data)

# Create scatter plots comparing normal distribution with others
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, size in enumerate(sample_sizes):
    normal_data = dataframes[size]['normal']
    for j, dist in enumerate(['t', 'uniform', 'gamma']):
        dist_data = dataframes[size][dist]
        axes[i, j].scatter(normal_data, dist_data)
        axes[i, j].set_title(f'Normal & {dist.capitalize()} (n={size})')
        axes[i, j].set_xlabel('Normal')
        axes[i, j].set_ylabel(dist.capitalize())

plt.tight_layout()
plt.show()

# Print the first few rows of each DataFrame
for size, df in dataframes.items():
    print(f"\nDataFrame for sample size {size}:")
    # print(df.head())
    df.to_csv(f'data/sim{size}.csv', index=False)

    omgd_result = omgd.omgd(df, Y=df.columns[0], factors=df.columns[1:], n_variates=1, disc_interval=range(3, 8))
    print(omgd_result['factor'])