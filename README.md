# Optimal Multivariate-stratification Geographical Detector
***Article DOI***: https://doi.org/10.1080/15481603.2024.2422941
***How to cite***: Guo, Y., Wu, Z., Zheng, Z., & Li, X. (2024). An optimal multivariate-stratification geographical detector model for revealing the impact of multi-factor combinations on the dependent variable. GIScience & Remote Sensing, 61(1). https://doi.org/10.1080/15481603.2024.2422941

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
Download anaconda from https://www.anaconda.com/, open anaconda promt (using search bar), change the dictonary to the folder which contains *'omgd.yml'*, input *'conda env create -f omgd.yml'* and *'conda env list'* to check if the omgd environment is configurated.
