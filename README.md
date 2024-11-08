# Optimal Multivariate-stratification Geographical Detector
##How to cite
Guo, Y., Wu, Z., Zheng, Z., & Li, X. (2024). An optimal multivariate-stratification geographical detector model for revealing the impact of multi-factor combinations on the dependent variable. GIScience & Remote Sensing, 61(1). https://doi.org/10.1080/15481603.2024.2422941

## Basic knowledge of Geodetector
http://geodetector.cn/

## Schematic diagram
![image](https://github.com/gisgyf/OMGD/blob/main/img/schematic%20diagram.png)

## Files description
- ***data*** folder contains all the case data used in the article, stored in csv file format.<br>
- ***omgd.py*** consists of the main functions of the Optimal Multivariate-stratification Geographical Detector, including computation functions and visualization functions.<br>
- ***sample.py*** is used to sample data at a certain ratio (e.g. 50%).<br>
- ***simulation.py*** offers simulation results discussed in Section 3.2.<br>
- ***test.ipynb*** and ***test.py*** are used to reproduce the results shown in the main text.<br>

## Python (Anaconda) Environment
You can create your own environment with the following dependencies or install the packages using pip, or download the conda environment(folder) directly and configurate it.
### Dependencies
- Python == 3.10
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
### Install using pip
> cd ***your folder path*** <br>
> pip install -r requirements.txt
### Conda environment
Download anaconda from https://www.anaconda.com/, open anaconda promt (using search bar), change the dictonary to the folder which contains ***'omgd.yml'***, input ***'conda env create -f omgd.yml'*** and ***'conda env list'*** to check if the omgd environment is configurated.

## How to use
Open ***test.ipynb*** or ***test.py***, run the code to see if it works.

## Description of main functions

### def scale_detector(path_list: Sequence, Y, factors:Sequence, disc_interval:Sequence, type_factors:Sequence=[], quantile:float=0.8, n_variates=1, random_state=0)

- scale detector can be used to detect the optimal spatial scale for spatial stratified heterogeneity analysis. <br>
- ***path_list*** is a list includes various files location (different spatial scales), ***Y*** is the dependent variable field name and ***factors*** is a list containing field name of explanatory variables. <br>
- ***disc_interval*** specifys the classification (stratification) number (e.g. [3, 7] indicates the explanatory variables / explanatory variables combinations are classified into 3, 4, 5, 6 and 7 categories iterately to find the optimal classification number. <br>
- ***type_factor*** specifys the fields (name) which is already categorized rather than continuous. <br>
- ***quantile*** defines how many variables are used to calculated the avaergae ***Q*** value (e.g., we have 10 variables / variables combinations, and the quantile is 0.8, then only the top 3 ***Q*** values are used to calculate the avaergae ***Q*** value). <br>
- ***n_variates*** indicates how many variables are used in the analysis. <br>
- the scale detector function returns scale_result (dataframe) and the best_scale (location of the optimal scale data file), the scale_result can be plotted using scale_plot (scale_result, size_list=[], dpi=100, unit='')

### def omgd(df:pd.DataFrame, Y, factors:Sequence, n_variates:int, disc_interval:Sequence, type_factors:Sequence=[], random_state=0)

- one step omgd model, returns a dictionary (omgd_result) contains original dataframe (omgd_result['original']), classification result (omgd_result['classify']), factor detector result (omgd_result['factor']), <br>
- interaction detector result (omgd_result['interaction']), risk detector result (omgd_result['risk']) and ecological detector result (omgd_result['ecological'])

### def omgd.factor_detector(df, Y, factors:Sequence)
### def omgd.interation_detector(df:pd.DataFrame, Y, factors:Sequence)
### def omgd.risk_detector(df, Y, factors:Sequence)
### def omgd.ecological_detector(df, Y, factors:Sequence)

- functions of four basic detectors. <br>
- ***df***: dataframe, includes ***Y*** fields (dependent variable) and ***factors*** fields (explanatory variables). <br>
- omgd.factor_detector returns a dataframe which is sorted according to Geodetector ***Q*** value. <br>
- omgd.interation_detector returns a dataframe which is sorted according to Geodetector ***Q*** value. <br>
- omgd.risk_detector returns a list, which includes the significance test results and the mean values of each strata. <br>
- omgd.ecological_detector returns a dataframe. <br>
- the four baisc detectors can be plotting using ***factor_plot***, ***factor_plot***, ***risk_plot*** and ***ecological_plot***, respectively. (e.g. omgd.factor_plot(factor_result)). <br>

### def classify(X, n_clusters, classify_result, colname, random_state=0)

- function classify is used to automatically classify (discritize) continuous variables or variables combinations into ***n_clusters*** catogory, ***X*** is the values dataframe (explanatory variables), the result is stored in ***classify_result*** and colname is made up of single or multiple fields (explanatory variables) spliced together using '_'. <br>
- the classification result can be plotted using 'classify_plot(original_df:pd.DataFrame, classify_df:pd.DataFrame, dpi=100, nrows=0, ncols=0, unit_list=[])'



