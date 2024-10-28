import pandas as pd
import omgd
import matplotlib.pyplot as plt


## parameters

# path_list = ['data/LST250.csv', 'data/LST500.csv', 'data/LST750.csv',
#              'data/LST1000.csv','data/LST1500.csv', 'data/LST2000.csv']
path_list = ['data/LST2000.csv', 'data/LST3000.csv', 'data/LST4000.csv', 'data/LST5000.csv',
             'data/LST6000.csv', 'data/LST7000.csv', 'data/LST8000.csv', 'data/LST9000.csv']
data = pd.read_csv('data/LST2000.csv')
Y = data.columns[0]
X = data.columns[1:]
discitv = range(3, 8)
n_variates = 2

# # scale detector
scale_result, best_scale = omgd.scale_detector(path_list, Y, X, discitv, quantile=0.8, n_variates=n_variates)
omgd.scale_plot(scale_result, size_list=[2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], dpi=200, unit='m')
plt.show()


## one step OMGD model

omgd_result = omgd.omgd(data, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv)
print(omgd_result['classify'])
# omgd_result['classify'].to_csv('LST_classify.csv')
print(omgd_result['factor'])
omgd.omgd_plot(omgd_result, unit_list=['°C', '%', 'm', '', '', '', ''])
plt.show()


## Discussion

# path_list = ['data/ndvi_5.csv', 'data/ndvi_10.csv', 'data/ndvi_20.csv', 'data/ndvi_30.csv', 'data/ndvi_40.csv', 'data/ndvi_50.csv']
# Y = 'NDVIchange'
# X = ['Climatezone', 'Mining', 'Tempchange', 'Precipitation', 'GDP', 'Popdensity']
# type_factors = ['Climatezone', 'Mining']
# discitv = range(3, 8)

# path_list = ['data/h1n1_50.csv', 'data/h1n1_100.csv', 'data/h1n1_150.csv']
# Y = 'H1N1'
# X = ['temp', 'prec', 'humi', 'popd', 'gdpd', 'rdds', 'sensepop', 'urbanpop', 'medicost', 'Georegion']
# type_factors = ['Georegion']
# discitv = range(3, 8)

# n_variates = 6
#
# scale_result, best_scale = omgd.scale_detector(path_list, Y, X, discitv, type_factors, quantile=0.8, n_variates=n_variates)
# omgd.scale_plot(scale_result, size_list=[5, 10, 20, 30, 40, 50], dpi=200, unit='km')
# plt.show()

# df = pd.read_csv('data/ndvi_40.csv')
# omgd_result = omgd.omgd(df, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=type_factors)
# omgd.factor_plot(omgd_result['factor'][:10])
#
# plt.show()


# path_list = ['data/PZH_LST1000.csv', 'data/PZH_LST2000.csv', 'data/PZH_LST3000.csv',
#              'data/PZH_LST4000.csv', 'data/PZH_LST5000.csv', 'data/PZH_LST6000.csv']
# data = pd.read_csv('data/PZH_LST5000.csv')
# Y = data.columns[0]
# X = data.columns[1:]
# discitv = range(3, 8)
# n_variates = 6
# #
# # # # scale detector
# # # scale_result, best_scale = omgd.scale_detector(path_list, Y, X, discitv, quantile=0.8, n_variates=n_variates)
# # # omgd.scale_plot(scale_result, size_list=[1, 2, 3, 4, 5, 6], dpi=200, unit='km')
# # # plt.show()
# #
# df = pd.read_csv('data/PZH_LST5000.csv')
# omgd_result = omgd.omgd(df, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv, type_factors=[])
# omgd.factor_plot(omgd_result['factor'][:10])
#
# plt.show()