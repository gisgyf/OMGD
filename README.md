# Optimal Multivariate-stratification Geographical Detector
***Article DOI***: https://doi.org/10.1080/15481603.2024.2422941
***How to cite***: Guo, Y., Wu, Z., Zheng, Z., & Li, X. (2024). An optimal multivariate-stratification geographical detector model for revealing the impact of multi-factor combinations on the dependent variable. GIScience & Remote Sensing, 61(1). https://doi.org/10.1080/15481603.2024.2422941

## Schematic diagram
[![pic](https://github.com/gisgyf/OMGD/blob/main/schematic%20diagram.png, "Schematic diagram")](https://github.com/gisgyf/OMGD/blob/main/schematic%20diagram.png)

## Files description
- ***data*** folder contains all the case data used in the article, stored in csv file format.<br>
- ***omgd.py*** consists of the main functions of the Optimal Multivariate-stratification Geographical Detector, including computation functions and visualization functions.<br>
- ***sample.py*** is used to sample data at a certain ratio (e.g. 50%).<br>
- ***simulation.py*** offers simulation results discussed in Section 3.2.<br>
- ***test.ipynb*** and ***test.py*** are used to reproduce the results shown in the main text.<br>

## Python(Anaconda) Environment
You can create your own environment with the following dependencies or download the conda environment(folder) directly and configurate it.
### Dependencies
- jenkspy == 0.4.0
- matplotlib == 3.7.3
- numpy == 2.6.4
- openpyxl == 3.0.10
- pandas == 2.1.4
- pyamg == 4.2.3
- pillow == 10.2.0
- scikit-learn == 1.3.0
- scipy == 1.12.0
- seaborn == 0.12.2
### Conda environment
Download anaconda from https://www.anaconda.com/, open anaconda promt (using search bar), change the dictonary to the folder which contains ***'omgd.yml'***, input ***'conda env create -f omgd.yml'*** and ***'conda env list'*** to check if the omgd environment is configurated.

## How to use
Open ***test.ipynb*** or ***test.py***, run the code to see if it works.
### Explanation
> path_list = ['data/LST2000.csv', 'data/LST3000.csv', 'data/LST4000.csv', 'data/LST5000.csv',
>              'data/LST6000.csv', 'data/LST7000.csv', 'data/LST8000.csv', 'data/LST9000.csv']
- Define a list that contains data with multiple spatial scale

> data = pd.read_csv('data/LST2000.csv')
> Y = data.columns[0]
> X = data.columns[1:]
> discitv = range(3, 8)
> n_variates = 2
- Define the dependent variable[Y] and explanatory variables[X], the discretization(classification) interval[discitv] and the number of explanatory variables used in the calculation(n_variates).

> scale_result, best_scale = omgd.scale_detector(path_list, Y, X, discitv, quantile=0.8, n_variates=n_variates)
> omgd.scale_plot(scale_result, size_list=[2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], dpi=200, unit='m')
> plt.show()
- Run the scale detector to detect the optimal spatial scale for spatial stratified heterogeneity
- ***Parameters:*** omgd.scale_detector(path_list: Sequence, Y, factors:Sequence, disc_interval:Sequence, type_factors:Sequence=[], quantile:float=0.8, n_variates=1, random_state=0)
- ***Parameters:*** omgd.scale_plot(scale_result, size_list=[], dpi=100, unit='')

> omgd_result = omgd.omgd(data, Y=Y, factors=X, n_variates=n_variates, disc_interval=discitv)
> print(omgd_result['classify'])
> # omgd_result['classify'].to_csv('LST_classify.csv')
> print(omgd_result['factor'])
> omgd.omgd_plot(omgd_result, unit_list=['°C', '%', 'm', '', '', '', ''])
> plt.show()
- One step OMGD model
- ***Parameters:*** omgd(df:pd.DataFrame, Y, factors:Sequence, n_variates:int, disc_interval:Sequence, type_factors:Sequence=[], random_state=0)
- The result of the ***omgd.omgd*** function returns a dictionary, defined as omgd_result here, which includes original dataframe: omgd_result['original'], classification result: omgd_result['classify'], result of the factor detector: omgd_result['factor'], result of the interaction detector: omgd_result['interaction'], result of the risk detector: omgd_result['risk'], and result of the ecological detector: omgd_result['ecological'].
