import pandas as pd
import numpy as np
from scipy.stats import ncf, t, f
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import jenkspy
from itertools import combinations, chain
from typing import Sequence
import matplotlib.pyplot as plt
import matplotlib.markers as mks
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import seaborn as sns
import math
import warnings
import os
warnings.filterwarnings("ignore")


def factor_detector(df, Y, factors:Sequence):
    y = df[Y]
    # calculate total number N and variance σ² of dependent variable
    N = len(y)
    sigma2 = np.var(y)
    factor_result = pd.DataFrame()

    for factor in factors:
        x = df[factor]
        # calculate q
        q = 1
        unique_x = np.unique(x)
        L = len(np.unique(x))
        if L == 1:
            continue
        group_means = []
        for val in unique_x:
            yi = y[x == val]
            group_means.append(yi.mean())
            ni = len(yi)
            sigma_i2 = np.var(yi)
            q -= (ni * sigma_i2) / (N * sigma2)

        # calculate p
        F = (N - L) / (L - 1) * q / (1 - q)
        group_means = np.array(group_means)
        λ = (1 / sigma2) * (np.sum(group_means ** 2) - (1 / N) *
                                  np.sum(np.sqrt([len(y[x == val]) for val in unique_x]) * group_means) ** 2)

        p = ncf.sf(F, L - 1, N - L, nc=λ)

        if p < 0.05:
            temp = pd.DataFrame({'factor': [factor], 'q': [q], 'p': [p]})
            factor_result = temp if factor_result.empty else pd.concat([factor_result, temp], ignore_index=True)

    if not factor_result.empty:
        factor_result = factor_result.sort_values(by='q', ascending=False).reset_index(drop=True)

    return factor_result

def interation_detector(df:pd.DataFrame, Y, factors:Sequence):
    df_copy = df.copy()
    # permutation
    pair_combinations = list(combinations(df[factors], 2))

    # interation factors
    interation_factors = []
    for combination in pair_combinations:
        c1, c2 = combination
        interation_factors.append(f'{c1}+{c2}')
        df_copy[f'{c1}+{c2}']= df_copy[c1].apply(str) + '_' + df_copy[c2].apply(str)
    interation_result = factor_detector(df_copy, Y, interation_factors)

    return interation_result

def factor_plot(factor_result, dpi=100):
    fig, ax = plt.subplots(dpi=dpi)

    factors = factor_result['factor']
    q = factor_result['q']

    colors = plt.cm.coolwarm_r(np.linspace(0.1, 0.9, len(factors)))
    ax.barh(factors, q, align='center', color=colors)

    for i, val in enumerate(q):
        ax.text(val, factors[i], f'{val:.4f}', ha='right', va='center')

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Q value')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()


def risk_detector(df:pd.DataFrame, Y, factors:Sequence):
    risk_result_list = []
    for factor in factors:
        # zonal statistics
        risk_result = df.groupby(factor)[Y].agg('mean').to_frame().T

        # Create a new DataFrame with the indexes of the column names of the original DataFrame, and all values are NaN
        df_new = pd.DataFrame(np.nan, index=risk_result.columns, columns=risk_result.columns)

        # Use the concat function to connect two DataFrames together.
        risk_result = pd.concat([risk_result, df_new])

        # permutation
        pair_combinations = list(combinations(set(df[factor]), 2))
        for combination in pair_combinations:
            # mean values
            mean_z1, mean_z2 = risk_result[combination[0]].values[0], risk_result[combination[1]].values[0]

            # dependent variables and its lengths
            y_z1, y_z2 = df[df[factor]==combination[0]][Y], df[df[factor]==combination[1]][Y]
            len_z1, len_z2 = len(y_z1), len(y_z2)
            if len_z1 <= 1 or len_z2 <= 1:
                risk_result.loc[combination[1], combination[0]] = False
                continue

            # sample variance
            variance_z1, variance_z2 = y_z1.var(ddof=1), y_z2.var(ddof=1)
            est_var_z1, est_var_z2 = variance_z1 / len_z1, variance_z2 / len_z2

            # t-statistics
            t_statistics = (mean_z1 - mean_z2) / np.sqrt(est_var_z1 + est_var_z2)
            # degrees of freedom
            if math.pow(est_var_z2, 2) / (len_z2 - 1) == 0:
                continue
            d_freedom = ((est_var_z1 + est_var_z2) /
                         (math.pow(est_var_z1, 2) / (len_z1 - 1) +
                          math.pow(est_var_z2, 2) / (len_z2 - 1)))
            # p
            p = t.sf(np.abs(t_statistics), d_freedom)

            # t-test
            risk_result.loc[combination[1], combination[0]] = True if p < 0.05 else False

        risk_result_list.append(risk_result)

    return risk_result_list

def risk_plot(risk_result_list, dpi=100, nrows=0, ncols=0, unit=''):
    if nrows==0 or ncols==0:
        nrows = math.ceil(math.sqrt(len(risk_result_list)))
        ncols = nrows - 1 if nrows * (nrows - 1) >= len(risk_result_list) else nrows

    fig, axes = plt.subplots(nrows, ncols, dpi=dpi)

    for i, val in enumerate(risk_result_list):
        if ncols == 1 and nrows == 1:
            temp_ax = axes
        else:
            temp_ax = axes[math.floor(i / ncols), i % ncols]

        risk_mean = val.copy().iloc[0]
        risk_sig = val.copy().iloc[1:]
        risk_sig.fillna(False, inplace=True)

        labels = risk_sig.astype(str)
        labels[labels == 'True'] = 'T'
        labels[labels == 'False'] = 'F'

        mask = np.triu(np.ones_like(risk_sig, dtype=bool))

        # check if is all True
        risk_sig = risk_sig.astype(int)
        total = sum(range(len(risk_sig)))
        cmap = ListedColormap(['#87CEEB', '#EE6363']) if risk_sig.sum().sum() < total else ListedColormap(['#EE6363'])
        sns.heatmap(risk_sig, mask=mask, annot=labels, fmt='', square=True, cbar=False, cmap=cmap, ax=temp_ax, annot_kws={'color': 'white'})
        # edge lines
        temp_ax.patch.set_visible(True)
        temp_ax.patch.set_edgecolor('black')
        temp_ax.patch.set_linewidth(0.8)

        temp_ax.tick_params(axis='y', labelrotation=0)
        temp_ax.set_xlabel('Class')
        temp_ax.set_ylabel('Class')
        temp_ax.set_xticks([j + 0.5 for j in risk_sig.index], list(risk_sig.index))
        temp_ax.set_yticks([j + 0.5 for j in risk_sig.index], list(risk_sig.index))
        temp_ax.set_title(val.columns.name)

        # Creating additional axes for plotting bar charts
        divider = make_axes_locatable(temp_ax)
        ax_bar = divider.append_axes("right", size="50%", pad=0)

        # same colors as classify result
        colors = plt.cm.rainbow(np.linspace(0, 1, len(risk_mean)))

        ax_bar.barh([j + 0.5 for j in risk_mean.index], risk_mean.values, color=colors, alpha=0.8)
        ax_bar.set_ylim(temp_ax.get_ylim())
        ax_bar.set_yticks([j + 0.5 for j in risk_mean.index], list(risk_mean.index))
        ax_bar.set_xticks([])
        ax_bar.set_xlabel(f'{val.index[0]} Mean') if unit == '' else ax_bar.set_xlabel(f'{val.index[0]} Mean ({unit})')

        for i2, val2 in enumerate(risk_mean.values):
            ax_bar.text(ax_bar.get_xlim()[0], i2 + 0.5, f'{val2:.2f}', ha='left', va='center')

    # remove the abundant axis
    if i < nrows * ncols - 1:
        for j in range(i + 1, nrows * ncols):
            axes[math.floor(j / ncols), j % ncols].set_axis_off()

    plt.tight_layout()


def ecological_detector(df:pd.DataFrame, Y, factors:Sequence):
    ecological_result = pd.DataFrame(columns=factors, index=factors)
    # permutation
    pair_combinations = list(combinations(factors, 2))

    for combination in pair_combinations:
        f1, f2 = combination

        # calculate variance of dependent variable under each independent variable
        sum_variance1, sum_variance2 = 0, 0
        N1, N2 = len(df[f1].notna()), len(df[f2].notna())

        for type in set(df[f1]):
            sum_variance1 += len(df[df[f1]==type].notna()) * np.var(df[df[f1]==type][Y])
        for type in set(df[f2]):
            sum_variance2 += len(df[df[f2]==type].notna()) * np.var(df[df[f2]==type][Y])

        # F and p
        if sum_variance1 > sum_variance2:
            F = (N2 * (N1 - 1) * sum_variance1) / (N1 * (N2 - 1) * sum_variance2)
            p = f.sf(F, N1 - 1, N2 - 1)
        else:
            F = (N2 * (N1-1) * sum_variance2) / (N1 * (N2-1) * sum_variance1)
            p = f.sf(F, N2 - 1, N1 - 1)

        ecological_result.loc[f2, f1] = True if p < 0.05 else False

    return ecological_result

def ecological_plot(ecological_result, dpi=100):
    df = ecological_result.copy()
    df.fillna(False, inplace=True)

    labels = df.astype(str)
    labels[labels == 'True'] = 'T'
    labels[labels == 'False'] = 'F'

    mask = np.triu(np.ones_like(df, dtype=bool))

    df = df.astype(int)
    total = sum(range(len(df)))

    fig, ax = plt.subplots(dpi=dpi)
    cmap = ListedColormap(['#87CEEB', '#EE6363']) if df.sum().sum() < total else ListedColormap(['#EE6363'])
    sns.heatmap(df, mask=mask, annot=labels, fmt='', square=True, cbar=False, cmap=cmap, ax=ax, annot_kws={'color': 'white'})

    # edge lines
    ax.patch.set_visible(True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(0.8)
    ax.set_xticklabels([i for i in range(len(df))], rotation='horizontal')
    ax.set_yticklabels([f'{i}: {val}' for i, val in enumerate(df.index)])

    plt.tight_layout()


def classify(X, n_clusters, classify_result, colname, random_state=0):
    if len(colname.split('_')) <= 2:
        # equal intervals
        uni = KBinsDiscretizer(n_bins=n_clusters, encode='ordinal', strategy='uniform')
        uni.fit(X)
        uni_disc = uni.transform(X)
        uni_disc = uni_disc.reshape(uni_disc.shape[0]).astype(int)
        uni_name = f'{colname}uni{n_clusters}'

        # quantile
        qt = KBinsDiscretizer(n_bins=n_clusters, encode='ordinal', strategy='quantile')
        qt.fit(X)
        qt_disc = qt.transform(X)
        qt_disc = qt_disc.reshape(qt_disc.shape[0]).astype(int)
        qt_name = f'{colname}qt{n_clusters}'

        # Jenks Natural Breaks
        jnb = jenkspy.JenksNaturalBreaks(n_clusters)
        jnb.fit(X.reshape(X.shape[0]))
        jnb_disc = jnb.labels_
        jnb_name = f'{colname}jnb{n_clusters}'

        # geometric interval
        gmt_std = (X.max() / X.min()) ** (1 / n_clusters)
        gmt_intervals = [X.min() * gmt_std ** k for k in range(n_clusters + 1)]
        gmt_disc = np.digitize(X, gmt_intervals)
        gmt_disc = list(chain.from_iterable(gmt_disc))
        gmt_name = f'{colname}gmt{n_clusters}'

        # standard deviation
        mean, min, max = (X.mean(), X.min(), X.max())
        std = np.std(X)
        std_intervals = [min, max]
        n = int(n_clusters / 2)
        count = 0
        for i in range(n):
            if i == 0 and n_clusters % 2 == 0:
                std_intervals.append(mean)
            else:
                if n_clusters % 2 != 0:
                    i += 1
                if mean - i * std < min:
                    std_intervals.append(mean + (i + count) * std)
                    count += 1
                    std_intervals.append(mean + (i + count) * std)
                elif mean + i * std > max:
                    std_intervals.append(mean - (i + count) * std)
                    count += 1
                    std_intervals.append(mean - (i + count) * std)
                else:
                    std_intervals.extend([mean - i * std, mean + i * std])

        std_intervals.sort()
        std_disc = np.digitize(X, std_intervals)
        std_disc = list(chain.from_iterable(std_disc))
        std_name = f'{colname}std{n_clusters}'

        disc_result = pd.DataFrame([uni_disc, qt_disc, jnb_disc, gmt_disc, std_disc]).T
        disc_result.columns = [uni_name, qt_name, jnb_name, gmt_name, std_name]
        classify_result = pd.concat([classify_result, disc_result], axis=1)

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # k-means
    if len(X) > 10000:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', init='k-means++')
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', init='k-means++')
    clusters_kms = kmeans.fit_predict(X)
    kms_name = f'{colname}kms{n_clusters}'

    # AgglomerativeClustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single', metric="euclidean")
    agg_clustering.fit(X)
    clusters_agg = agg_clustering.labels_
    agg_name = f'{colname}agg{n_clusters}'

    # SpectralClustering
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='amg', random_state=random_state,
                                             affinity='nearest_neighbors', n_jobs=-1, assign_labels='cluster_qr')
    spectral_clustering.fit(X)
    clusters_spc = spectral_clustering.labels_
    spc_name = f'{colname}spc{n_clusters}'

    # GaussianMixture
    gsm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gsm.fit(X)
    clusters_gsm = gsm.predict(X)
    gsm_name = f'{colname}gsm{n_clusters}'

    # BisectingKMeans
    bisecting_kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++',
                                       bisecting_strategy='largest_cluster')
    bisecting_kmeans.fit(X)
    clusters_bkm = bisecting_kmeans.labels_
    bkm_name = f'{colname}bkm{n_clusters}'

    # set new columns based on the clustering results
    clusters = pd.DataFrame([clusters_kms, clusters_agg, clusters_spc, clusters_gsm, clusters_bkm]).T
    clusters.columns = [kms_name, agg_name, spc_name, gsm_name, bkm_name]
    classify_result = pd.concat([classify_result, clusters], axis=1)

    return classify_result

def classify_plot(original_df:pd.DataFrame, classify_df:pd.DataFrame, dpi=100, nrows=0, ncols=0, unit_list=[]):
    Y = classify_df.columns[0]
    cols2split = classify_df.columns[1:]
    final_cols = [[j for j in i.split('_')] for i in classify_df[cols2split]]
    nvars = len(final_cols[0])
    unit_dict = {}
    if len(unit_list) == len(original_df.columns):
        unit_dict = {key: value for key, value in zip(original_df.columns, unit_list)}

    if nrows==0 or ncols==0:
        nrows = math.ceil(math.sqrt(len(cols2split)))
        ncols = nrows - 1 if nrows * (nrows - 1) >= len(cols2split) else nrows

    # if number of variables >= 2, 3d plot, else 2d plot
    if nvars > 4:
        return np.nan
    elif nvars > 3:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={"projection": "3d"}, dpi=dpi)
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=dpi)

    # plot in subplots
    for i in range(nrows):
        for j in range(ncols):
            plot_index = i*ncols+j
            if ncols == 1 and nrows == 1:
                temp_ax = axes
            else:
                temp_ax = axes[math.floor(plot_index / ncols), plot_index % ncols]
            if plot_index >= len(final_cols):
                temp_ax.set_axis_off()
                continue

            x_label = final_cols[plot_index][0]
            y_label = Y if nvars <= 2 else final_cols[plot_index][1]
            z_label = final_cols[plot_index][2] if nvars > 3 else ''

            if z_label:
                scatter = temp_ax.scatter(original_df[x_label], original_df[y_label], original_df[z_label], s=7,
                                   c=classify_df[classify_df.columns[plot_index+1]], cmap='rainbow', alpha=0.8)
                temp_ax.set_zlabel(z_label)
            else:
                scatter = temp_ax.scatter(original_df[x_label], original_df[y_label], s=7,
                                   c=classify_df[classify_df.columns[plot_index + 1]], cmap='rainbow', alpha=0.8)

            temp_ax.set_xlabel(x_label)
            temp_ax.set_ylabel(y_label)
            if unit_dict != {}:
                if unit_dict[x_label] != '':
                    temp_ax.set_xlabel(f'{x_label} ({unit_dict[x_label]})')
                if unit_dict[y_label] != '':
                    temp_ax.set_ylabel(f'{y_label} ({unit_dict[y_label]})')

            # Creating additional axes for plotting bar charts
            divider = make_axes_locatable(temp_ax)
            if nvars > 3:
                ax_legend = divider.append_axes("left", size="30%", pad=0, axes_class=plt.Axes)
            else:
                ax_legend = divider.append_axes("right", size="30%", pad=0, axes_class=plt.Axes)
            legend = ax_legend.legend(*scatter.legend_elements(), loc='center', framealpha=0,
                                       ncol=math.ceil(classify_df[classify_df.columns[plot_index + 1]].nunique() / 10))
            ax_legend.add_artist(legend)
            ax_legend.set_xticks([])
            ax_legend.set_yticks([])
            ax_legend.set_ylabel(classify_df.columns[plot_index+1].split('_')[-1] + '_class')

    plt.tight_layout()


def omgd(df:pd.DataFrame, Y, factors:Sequence, n_variates:int, disc_interval:Sequence, type_factors:Sequence=[], random_state=0):
    # deal with type factors:
    for type_factor in type_factors:
        # get unique
        unique_items = set(df[type_factor])
        # reclass the type factor to number
        mapping_dict = {}
        for i, item in enumerate(sorted(unique_items)):
            mapping_dict[item] = i
        df[type_factor] = [mapping_dict[item] for item in df[type_factor]]
        print(f'transform of {type_factor}: \n {mapping_dict}')

    # permutation
    pair_combinations = list(combinations(df[factors], n_variates))
    # Predefined classification results
    classify_result = df[Y].copy()
    factor_result = pd.DataFrame()

    for i in pair_combinations:
        # Optimization of each combination except type factors
        temp_result = df[Y].copy()

        if len(i) == 1 and i[0] in type_factors:
            temp_result = pd.concat([temp_result, df[i[0]]], axis=1)
            optimal = factor_detector(temp_result, Y, temp_result.columns[1:])
            classify_result = pd.concat([classify_result, temp_result[temp_result.columns[1:]]], axis=1)
            if not optimal.empty:
                # transform into dataframe
                optimal = optimal.loc[0].to_frame().T
                factor_result = optimal if factor_result.empty else pd.concat([factor_result, optimal])
            continue

        X = df[list(i)].values

        basic_name = ''
        for k in range(len(i)):
            basic_name = list(i)[k] + '_' if basic_name == '' else basic_name + list(i)[k] + '_'

        for j in disc_interval:
            temp_result = classify(X, j, temp_result, basic_name, random_state)

        classify_result = pd.concat([classify_result, temp_result[temp_result.columns[1:]]], axis=1)
        optimal = factor_detector(temp_result, Y, temp_result.columns[1:])
        if not optimal.empty:
            # transform into dataframe
            optimal = optimal.loc[0].to_frame().T
            factor_result = optimal if factor_result.empty else pd.concat([factor_result, optimal])

    # result store in dictionary
    omgd_result = {}
    omgd_result['original'] = df

    classify_result = classify_result[[Y] + list(factor_result['factor'])]
    omgd_result['classify'] = classify_result

    factor_result = factor_result.sort_values(by='q', ascending=False).reset_index(drop=True)
    omgd_result['factor'] = factor_result

    if n_variates == 1:
        interaction_result = interation_detector(classify_result, Y, classify_result.columns[1:])
        omgd_result['interaction'] = interaction_result

    risk_result_list = risk_detector(classify_result, Y, classify_result.columns[1:])
    omgd_result['risk'] = risk_result_list
    if n_variates < len(factors):
        ecological_result = ecological_detector(classify_result, Y, classify_result.columns[1:])
        omgd_result['ecological'] = ecological_result

    return omgd_result

def omgd_plot(omgd_result, dpi=100, nrows=0, ncols=0, unit_list:list=[]):
    classify_plot(omgd_result['original'], omgd_result['classify'], dpi=dpi, nrows=nrows, ncols=ncols, unit_list=unit_list)
    factor_plot(omgd_result['factor'], dpi=dpi)
    if 'interaction' in omgd_result.keys():
        factor_plot(omgd_result['interaction'], dpi=dpi)
    risk_plot(omgd_result['risk'], dpi=dpi, nrows=nrows, ncols=ncols, unit=unit_list[0] if len(unit_list) > 0 else '')
    if 'ecological' in omgd_result.keys():
        ecological_plot(omgd_result['ecological'], dpi=dpi)


def scale_detector(path_list: Sequence, Y, factors:Sequence, disc_interval:Sequence, type_factors:Sequence=[], quantile:float=0.8, n_variates=1, random_state=0):
    scale_result = pd.DataFrame()

    for path in path_list:
        base_name = os.path.basename(path).split('.')[0]
        if path.endswith('csv'):
            df = pd.read_csv(path)
        elif path.endswith('xls') or path.endswith('xlsx'):
            df = pd.read_excel(path)
        else:
            print(f'{path} has invalid file type')
            continue

        # deal with type factors:
        for type_factor in type_factors:
            # get unique
            unique_items = set(df[type_factor])
            # reclass the type factor to number
            mapping_dict = {}
            for i, item in enumerate(sorted(unique_items)):
                mapping_dict[item] = i
            df[type_factor] = [mapping_dict[item] for item in df[type_factor]]

        # permutation
        pair_combinations = list(combinations(df[factors], n_variates))
        # Predefined classification results
        factor_result = pd.DataFrame()

        for i in pair_combinations:
            # Optimization of each combination except type factors
            temp_result = df[Y].copy()

            if len(i) == 1 and i[0] in type_factors:
                temp_result = pd.concat([temp_result, df[i[0]]], axis=1)
                optimal = factor_detector(temp_result, Y, temp_result.columns[1:])
                if not optimal.empty:
                    # transform into dataframe
                    optimal = optimal.loc[0].to_frame().T
                    factor_result = optimal if factor_result.empty else pd.concat([factor_result, optimal])
                continue

            X = df[list(i)].values

            basic_name = ''
            for k in range(len(i)):
                basic_name = list(i)[k] + '_' if basic_name == '' else basic_name + list(i)[k] + '_'

            for j in disc_interval:
                temp_result = classify(X, j, temp_result, basic_name, random_state)

            optimal = factor_detector(temp_result, Y, temp_result.columns[1:])
            if not optimal.empty:
                # transform into dataframe
                optimal = optimal.loc[0].to_frame().T
                factor_result = optimal if factor_result.empty else pd.concat([factor_result, optimal])

        factor_result = factor_result[['factor', 'q']].rename(columns={'q': f'{base_name}'})
        factor_result['factor'] = factor_result['factor'].apply(lambda x: x if '_' not in x else '_'.join(x.split('_')[:-1]))
        scale_result = factor_result if scale_result.empty else scale_result.merge(factor_result, on='factor', how='outer')

    scale_result = scale_result.set_index('factor')
    scale_result = scale_result.apply(pd.to_numeric)

    # Calculate the average of the selected quantile Q values in each column
    quantiles = scale_result.quantile(quantile)
    evaluate = scale_result.apply(lambda col: col[col >= quantiles[col.name]].mean())
    scale_result.loc[f'{quantile:.0%} quantile'] = evaluate
    best_scale = path_list[list(evaluate).index(max(evaluate))]

    return scale_result, best_scale

def scale_plot(scale_result, size_list=[], dpi=100, unit=''):
    # plot the result
    x = scale_result.index
    colors_list = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    markers_list = list(mks.MarkerStyle.markers.keys())
    labels_line = []

    # plot factors values lines
    fig, ax1 = plt.subplots(dpi=dpi)
    for i, val in enumerate(x[:-1]):
        line = ax1.plot(scale_result.columns, scale_result.loc[val], color=colors_list[i],
                        marker=markers_list[i], linestyle='--', label=val)
        labels_line.extend(line)

    # plot quantile values line
    ax2 = ax1.twinx()
    quantile_values = scale_result.loc[x[-1]]
    final_line = ax2.plot(scale_result.columns, quantile_values, color='black', marker='X',
                          linewidth=2, markersize=12, label=x[-1])
    labels_line.extend(final_line)

    # add text for quantile values
    for i, val in enumerate(quantile_values):
        ax2.annotate(f'{val:.4f}', (scale_result.columns[i], quantile_values[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    # add legend, axis label and show the plot
    labels = [l.get_label() for l in labels_line]
    plt.legend(labels_line, labels, loc='lower right', framealpha=0, ncol=math.ceil(len(scale_result) / 10))
    if size_list != []:
        ax1.set_xticks(scale_result.columns, size_list)
    ax1.set_xlabel(f'Size of spatial unit ({unit})') if unit != '' else ax1.set_xlabel('Size of spatial unit')
    ax1.set_ylabel('Q value')
    ax2.set_ylabel(f'The {labels[-1]} of Q values')

    plt.tight_layout()
