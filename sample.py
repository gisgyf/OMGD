import pandas as pd
import omgd

data = pd.read_csv('data/LST2000.csv')
Y = data.columns[0]
X = data.columns[1:]
discitv = range(3, 8)
n_variates = 1
type_factors = []

# sampling of 25%, 50% and 75% of data
data25 = data.sample(frac=0.25, random_state=0).reset_index()
data50 = data.sample(frac=0.50, random_state=0).reset_index()
data75 = data.sample(frac=0.75, random_state=0).reset_index()


omgd25 = omgd.omgd(data25, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=type_factors)
print('omgd25\n', omgd25['factor'])
omgd50 = omgd.omgd(data50, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=type_factors)
print('omgd50\n', omgd50['factor'])
omgd75 = omgd.omgd(data75, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=type_factors)
print('omgd75\n', omgd75['factor'])
omgd100 = omgd.omgd(data, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=type_factors)
print('omgd100\n', omgd100['factor'])