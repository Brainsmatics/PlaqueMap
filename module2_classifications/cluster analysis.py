'''

Cluster analysis

Comprising two parts: model training and prediction.
During training, a comprehensive selection can be made through four cluster number selection methods.

An example of the operation can be found in the main function.
Comprising four feature groups: Gray, Shape, Texture, Multi.

'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn.decomposition import PCA
import pickle
import joblib
import os
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


# pick
feature_multi = ["ID", "data", "num", "age_month", "r", "cls", "max_gray",
               "value_min", "ratio", "volume_regist", "shape_VoxelVolume", "shape_MeshVolume", "shape_SurfaceArea",
               "shape_SurfaceVolumeRatio", "shape_Compactness1", "shape_Compactness2", "shape_Sphericity",
               "shape_SphericalDisproportion",
               "shape_Maximum3DDiameter", "shape_Maximum2DDiameterSlice", "shape_Maximum2DDiameterColumn",
               "shape_Maximum2DDiameterRow", "shape_MajorAxisLength", "shape_MinorAxisLength", "shape_LeastAxisLength",
               "shape_Elongation", "shape_Flatness", "glcm_Autocorrelation", "glcm_JointAverage",
               "glcm_ClusterProminence",
               "glcm_ClusterShade", "glcm_ClusterTendency", "glcm_Contrast", "glcm_Correlation",
               "glcm_DifferenceAverage",
               "glcm_DifferenceEntropy", "glcm_DifferenceVariance", "glcm_JointEnergy", "glcm_JointEntropy",
               "glcm_Imc1",
               "glcm_Imc2", "glcm_Idm", "glcm_Idmn", "glcm_Id", "glcm_Idn", "glcm_InverseVariance",
               "glcm_MaximumProbability",
               "glcm_SumEntropy", "glcm_SumSquares", "glszm_GrayLevelVariance", "glszm_ZoneVariance",
               "glszm_SizeZoneNonUniformity",
               "glszm_GrayLevelNonUniformity", "glszm_SmallAreaHighGrayLevelEmphasis",
               "glszm_LargeAreaLowGrayLevelEmphasis",
               "glszm_LargeAreaHighGrayLevelEmphasis", "glszm_SmallAreaLowGrayLevelEmphasis", "glszm_SmallAreaEmphasis",
               "glszm_LargeAreaEmphasis", "glszm_GrayLevelNonUniformityNormalized",
               "glszm_SizeZoneNonUniformityNormalized",
               "glszm_ZonePercentage", "glszm_ZoneEntropy", "glszm_LowGrayLevelZoneEmphasis",
               "glszm_HighGrayLevelZoneEmphasis",
               "ngtdm_Coarseness", "ngtdm_Contrast", "ngtdm_Busyness", "ngtdm_Complexity", "ngtdm_Strength",
               "gldm_SmallDependenceEmphasis",
               "gldm_LargeDependenceEmphasis", "gldm_GrayLevelNonUniformity", "gldm_DependenceNonUniformity",
               "gldm_DependenceNonUniformityNormalized", "gldm_GrayLevelVariance", "gldm_DependenceVariance",
               "gldm_DependenceEntropy",
               "gldm_LowGrayLevelEmphasis", "gldm_HighGrayLevelEmphasis", "gldm_SmallDependenceLowGrayLevelEmphasis",
               "gldm_SmallDependenceHighGrayLevelEmphasis", "gldm_LargeDependenceLowGrayLevelEmphasis",
               "gldm_LargeDependenceHighGrayLevelEmphasis", "region"]
feature_shape = ["ID", "data", "num", "age_month", "r", "cls",
                 "value_min", "ratio", "volume_regist", "shape_VoxelVolume", "shape_MeshVolume", "shape_SurfaceArea",
                 "shape_SurfaceVolumeRatio", "shape_Compactness1", "shape_Compactness2", "shape_Sphericity",
                 "shape_SphericalDisproportion",
                 "shape_Maximum3DDiameter", "shape_Maximum2DDiameterSlice", "shape_Maximum2DDiameterColumn",
                 "shape_Maximum2DDiameterRow", "shape_MajorAxisLength", "shape_MinorAxisLength",
                 "shape_LeastAxisLength",
                 "shape_Elongation", "shape_Flatness", "glszm_ZoneVariance", "glszm_SizeZoneNonUniformity",
                 "glszm_SizeZoneNonUniformityNormalized", "region"]
feature_gray = ["ID", "data", "num", "age_month", "cls", "max_gray",
                "glcm_JointAverage", "glcm_ClusterProminence", "glcm_ClusterShade", "glcm_ClusterTendency",
                "glcm_Contrast", "glcm_Correlation", "glcm_DifferenceAverage",
                "glcm_DifferenceEntropy", "glcm_DifferenceVariance", "glcm_JointEnergy",
                # "glcm_JointEntropy"(this feature is missing from the gray model)
                "glcm_Idm", "glcm_Idmn", "glcm_Id", "glcm_Idn",
                "glcm_InverseVariance", "glcm_MaximumProbability", "glcm_SumEntropy", "glcm_SumSquares",
                "glszm_GrayLevelVariance", "glszm_GrayLevelNonUniformity", "glszm_SmallAreaHighGrayLevelEmphasis",
                "glszm_LargeAreaLowGrayLevelEmphasis", "glszm_LargeAreaHighGrayLevelEmphasis",
                "glszm_SmallAreaLowGrayLevelEmphasis",
                "glszm_GrayLevelNonUniformityNormalized", "glszm_LowGrayLevelZoneEmphasis",
                "glszm_HighGrayLevelZoneEmphasis",
                "ngtdm_Contrast", "ngtdm_Busyness", "ngtdm_Complexity", "ngtdm_Strength",
                "gldm_GrayLevelNonUniformity", "gldm_DependenceNonUniformity",
                "gldm_DependenceNonUniformityNormalized", "gldm_GrayLevelVariance", "gldm_DependenceVariance",
                "gldm_DependenceEntropy",
                "gldm_LowGrayLevelEmphasis", "gldm_HighGrayLevelEmphasis", "gldm_SmallDependenceLowGrayLevelEmphasis",
                "gldm_SmallDependenceHighGrayLevelEmphasis", "gldm_LargeDependenceLowGrayLevelEmphasis",
                "gldm_LargeDependenceHighGrayLevelEmphasis", "region"]
feature_texture = ["ID", "data", "num", "age_month", "cls",
                   "glcm_Autocorrelation", "glcm_Imc1", "glcm_Imc2", "glszm_SmallAreaEmphasis",
                   "glszm_LargeAreaEmphasis", "glszm_ZonePercentage", "glszm_ZoneEntropy", "ngtdm_Coarseness",
                   "gldm_SmallDependenceEmphasis", "gldm_LargeDependenceEmphasis", "region"]


def shouzhou(df_normalized_data, out_dir):
    # The elbow method identifies the appropriate k-cluster by locating the inflection point on the plot corresponding to k.

    plt.style.use('seaborn')
    plt.rcParams['axes.unicode_minus'] = False
    K=range(1,200)

    sse_result = []
    L = []
    for k in K:
        kms_plus = KMeans(n_clusters=k, init='k-means++')
        kms_plus.fit(df_normalized_data)
        # sse
        sse_result.append(sum(np.min(cdist(df_normalized_data, kms_plus.cluster_centers_, 'euclidean'), axis=1))
                          / df_normalized_data.shape[0])
        # inertia_
        L.append((k, kms_plus.inertia_))
    sz_sse = DataFrame(sse_result)
    sz_sse.to_excel(out_dir+'sz-sse.xlsx')
    a = pd.DataFrame(L)
    a.columns = ['k', 'inertia']
    a.to_excel(out_dir+'sz-iter.xlsx')

    plt.figure(figsize=(50, 7))
    plt.subplot(1, 2, 1)
    plt.plot(K, sse_result, 'gp-')
    plt.xlabel('k', fontsize=20)
    plt.ylabel(u'Average degree of distortion')
    plt.title(u'Best k (sse)', fontdict={'fontsize': 15})
    plt.xticks(K)
    plt.subplot(1, 2, 2)
    plt.plot(a.k, a.inertia, 'gp-')
    plt.xlabel('k', fontsize=20)
    plt.ylabel('inertia')
    plt.title(u'best k (inertia)', fontdict={'fontsize': 15})
    plt.xticks(a.k)
    # method 2 : inertia_
    # L = []
    # for k in K:
    #     kms_plus = KMeans(n_clusters=k, init='k-means++')
    #     kms_plus.fit(df_normalized_data)
    #     L.append((k, kms_plus.inertia_))
    # a = pd.DataFrame(L)
    # a.columns = ['k', 'inertia']
    # a.to_excel('./output/sz-iter.xlsx')
    # plt.subplot(1, 2, 2)
    # plt.plot(a.k, a.inertia, 'gp-')
    # plt.xlabel('k', fontsize=20)
    # plt.ylabel('inertia')
    # plt.title(u'best k (2)', fontdict={'fontsize': 15})
    # plt.xticks(a.k)
    os.makedirs('./output/figs', exist_ok=True)
    plt.savefig("./output/figs/sz.png")



def contour(df_normalized_data, out_dir):
    # The cluster with the largest contour coefficient is the most suitable.
    plt.style.use('seaborn')
    plt.rcParams['axes.unicode_minus'] = False

    L = []
    for k in range(2, 200):
        kms_plus = KMeans(n_clusters=k, init='k-means++')
        kms_plus.fit(df_normalized_data)
        L.append([k, silhouette_score(df_normalized_data, kms_plus.labels_)])
    a = pd.DataFrame(L)
    a.columns = ['k', 'contour']
    a.to_excel(out_dir+'lunkuo.xlsx')
    index = a['contour'].idxmax()
    K_ = a.iloc[index]['k']
    max_ = a.iloc[index]['contour']

    plt.figure(figsize=(8, 5), dpi=70)
    plt.plot(a.k, a.contour, 'gp-')
    plt.xticks(a.k)
    plt.xlabel('k')
    plt.ylabel('contour')
    plt.title('contour best k', fontdict={'fontsize': 15})
    os.makedirs(out_dir + 'figs', exist_ok=True)
    plt.savefig(out_dir+'figs/contour.png')
    # plt.show()
    return K_, max_


def CH(df_normalized_data, out_dir):
    # Calinski-Harabasz coefficient for k determination.
    plt.style.use('seaborn')
    plt.rcParams['axes.unicode_minus'] = False

    L = []
    for k in range(2, 200):
        kms_plus = KMeans(n_clusters=k, init='k-means++')
        kms_plus.fit(df_normalized_data)
        L.append([k, metrics.calinski_harabasz_score(df_normalized_data, kms_plus.labels_)])
    a = pd.DataFrame(L)
    a.columns = ['k', 'Calinski-Harabasz']
    os.makedirs('./output', exist_ok=True)
    a.to_excel('./output/Calinski-Harabasz.xlsx')
    index = a['Calinski-Harabasz'].idxmax()
    K_CH = a.iloc[index]['k']
    max_CH = a.iloc[index]['Calinski-Harabasz']

    plt.figure(figsize=(8, 5), dpi=70)
    plt.plot(a["k"], a["Calinski-Harabasz"], 'gp-')
    plt.xticks(a.k)
    plt.xlabel('k')
    plt.ylabel('Calinski-Harabasz')
    plt.title('Calinski-Harabasz best k', fontdict={'fontsize': 15})
    os.makedirs(out_dir + 'figs', exist_ok=True)
    plt.savefig(out_dir+'figs/CH.png')
    # plt.show()
    return K_CH, max_CH


def DB(df_normalized_data, out_dir):
    # The Davies-Bouldin coefficient is used to determine the value of k.
    plt.style.use('seaborn')
    plt.rcParams['axes.unicode_minus'] = False

    L = []
    for k in range(2, 200):
        kms_plus = KMeans(n_clusters=k, init='k-means++')
        kms_plus.fit(df_normalized_data)
        L.append([k, metrics.davies_bouldin_score(df_normalized_data, kms_plus.labels_)])
    a = pd.DataFrame(L)
    a.columns = ['k', 'davies-bouldin']
    a.to_excel(out_dir+'davies-bouldin.xlsx')
    index = a['davies-bouldin'].idxmin()
    K_DB = a.iloc[index]['k']
    max_DB = a.iloc[index]['davies-bouldin']

    plt.figure(figsize=(8, 5), dpi=70)
    plt.plot(a["k"], a["davies-bouldin"], 'gp-')
    plt.xticks(a.k)
    plt.xlabel('k')
    plt.ylabel('davies-bouldin')
    plt.title('davies-bouldin best k', fontdict={'fontsize': 15})
    os.makedirs(out_dir + 'figs', exist_ok=True)
    plt.savefig(out_dir+'figs/DB.png')
    # plt.show()
    return K_DB, max_DB



def sne_show(data, r, k, out_dir):
    tsne = TSNE()
    tsne.fit_transform(data)
    tsne = pd.DataFrame(tsne.embedding_, index=data.index)

    plt.rcParams['axes.unicode_minus'] = False

    colors = ['OrangeRed', 'Coral', 'LightSalmon', 'SandyBrown', 'DarkOrange', 'Goldenrod',
              'Gold', 'Yellow', 'Olive', 'DarkGreen', 'ForestGreen', 'LimeGreen',
              'LightGreen', 'Turquoise', 'Teal', 'DarkSlateGray', 'CadetBlue', 'SteelBlue',
              'Navy', 'DarkSlateBlue', 'MediumPurple']
    colors_ = ['DarkGreen', 'SteelBlue', 'OrangeRed']  # three classes
    # colors_id = ['#FF4500', '#FF7F50', '#FFA07A', '#F4A460', '#FF8C00', '#DAA520',
    #              '#FFD700', '#FFFF00', '#808000', '#006400', '#228B22', '#32CD32',
    #              '#90EE90', '#40E0D0', '#008080', '#2F4F4F', '#5F9EA0', '#4682B4',
    #              '#000080', '#483D8B', '#9370DB']

    plt.figure(figsize=(50, 50))
    # plt.figure(figsize=(15, 15))
    d_all = pd.DataFrame()
    for i in range(k):
        d = tsne[r[u'label'] == i]
        plt.plot(d[0], d[1], marker="o", color=colors[i], markersize=4, linestyle='')
        # plt.plot(d[0], d[1], marker="o", color=colors_[i], markersize=15, linestyle='')
        d_single = pd.DataFrame(d)
        d_all = pd.concat([d_all, d_single])
    # d = tsne[r[u'label'] == 0]
    # plt.plot(d[0], d[1], 'r.')
    # d = tsne[r[u'label'] == 1]
    # plt.plot(d[0], d[1], 'go')
    # d = tsne[r[u'label'] == 2]
    # plt.plot(d[0], d[1], 'b*')
    # d = tsne[r[u'label'] == 3]
    # plt.plot(d[0], d[1], 'yd')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    plt.tick_params(width=4, length=10, pad=5, labelsize=20)
    os.makedirs(out_dir + 'figs', exist_ok=True)
    plt.savefig(out_dir+'figs/sne-'+str(k)+'.png', dpi=600, transparent=True)
    d_all.to_csv(out_dir+'figs/sne-d.csv')
    # plt.show()


def sne_show_region(data, r, k, e_e, r_s, t, out_dir):
    tsne = TSNE(early_exaggeration=e_e, random_state=r_s)
    tsne.fit_transform(data)
    tsne = pd.DataFrame(tsne.embedding_, index=data.index)

    plt.rcParams['axes.unicode_minus'] = False

    colors = ['OrangeRed', 'Coral', 'LightSalmon', 'SandyBrown', 'DarkOrange', 'Goldenrod',
              'Gold', 'Yellow', 'Olive', 'DarkGreen', 'ForestGreen', 'LimeGreen',
              'LightGreen', 'Turquoise', 'Teal', 'DarkSlateGray', 'CadetBlue', 'SteelBlue',
              'Navy', 'DarkSlateBlue', 'MediumPurple']
    colors_ = ['DarkGreen', 'SteelBlue', 'OrangeRed']  # three classes
    # colors_id = ['#FF4500', '#FF7F50', '#FFA07A', '#F4A460', '#FF8C00', '#DAA520',
    #              '#FFD700', '#FFFF00', '#808000', '#006400', '#228B22', '#32CD32',
    #              '#90EE90', '#40E0D0', '#008080', '#2F4F4F', '#5F9EA0', '#4682B4',
    #              '#000080', '#483D8B', '#9370DB']

    for region in range(1,16):
        plt.figure(figsize=(50, 50))
        d_all = pd.DataFrame()
        for i in range(k):
            d = tsne[(r[u'label'] == i) & (r[u'region_big'] == region)]
            plt.plot(d[0], d[1], marker="o", color=colors[i], markersize=4, linestyle='')
            # plt.plot(d[0], d[1], marker="o", color=colors_[i], markersize=15, linestyle='')
            d_single = pd.DataFrame(d)
            d_all = pd.concat([d_all, d_single])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['top'].set_linewidth(4)
        plt.tick_params(width=4, length=10, pad=5, labelsize=20)
        os.makedirs(out_dir + 'figs', exist_ok=True)
        plt.savefig(out_dir+'figs/region'+str(region)+"-"+str(k)+'-'+str(t)+'.png', transparent=True, dpi=300)   # t denotes the test round; output multiple times for testing
        d_all.to_csv(out_dir+'figs/region'+str(region)+'.csv')
        plt.close()

    for cls in range(1, 4):
        plt.figure(figsize=(50, 50))
        d_all = pd.DataFrame()
        for i in range(k):
            d = tsne[(r[u'label'] == i) & (r[u'cls'] == cls)]
            plt.plot(d[0], d[1], marker="o", color=colors[i], markersize=4, linestyle='')
            # plt.plot(d[0], d[1], marker="o", color=colors_[i], markersize=15, linestyle='')
            d_single = pd.DataFrame(d)
            d_all = pd.concat([d_all, d_single])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['top'].set_linewidth(4)
        plt.tick_params(width=4, length=10, pad=5, labelsize=20)
        plt.savefig(out_dir + 'figs/cls' + str(cls) + "-" + str(k) + '-' + str(t) + '.png', transparent=True, dpi=300)
        d_all.to_csv(out_dir + 'figs/cls' + str(cls) + '.csv')
        plt.close()

    plt.figure(figsize=(50, 50))
    d_all = pd.DataFrame()
    for i in range(k):
        d = tsne[r[u'label'] == i]
        plt.plot(d[0], d[1], marker="o", color=colors[i], markersize=4, linestyle='')
        # plt.plot(d[0], d[1], marker="o", color=colors_[i], markersize=15, linestyle='')
        d_single = pd.DataFrame(d)
        d_all = pd.concat([d_all, d_single])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    plt.tick_params(width=4, length=10, pad=5, labelsize=20)
    plt.savefig(out_dir + 'figs/sne-' + str(k) + '-' + str(t) + '.png', dpi=300, transparent=True)
    d_all.to_csv(out_dir + 'figs/sne-d.csv')
    plt.close()



def convert_string_to_complex(series):
    """
    Convert strings in a pandas Series to complex numbers.
    """

    def convert_value(x):
        if pd.isna(x):
            return np.nan
        elif isinstance(x, (int, float, complex)):
            return complex(x)
        elif isinstance(x, str):
            try:
                s = x.strip().replace(' ', '').replace('i', 'j')
                return complex(s)
            except (ValueError, TypeError):
                return np.nan
        else:
            return np.nan
    return series.apply(convert_value)


def process_dataframe_complex_columns(df, columns):
    """
    Processing complex values in specified columns of a DataFrame.

    df: DataFrame
    columns: The list of column names to process; if None, all numeric columns shall be processed.
    """
    if columns is None:
        # Automatically detect columns that may contain complex values.
        columns = df.select_dtypes(include=[object, complex]).columns
    result_df = df.copy()
    for col in columns:
        result_df[col] = result_df[col].apply(
            # lambda x: x[1:-1] if isinstance(x, str) and len(x) > 2 else x
            lambda x: x[1:-1] if isinstance(x, str) else x
        )
        result_df[col] = convert_string_to_complex(result_df[col])
        result_df[col] = result_df[col].apply(
            lambda x: abs(x) if isinstance(x, complex) else float(x)
        )
    return result_df


def norm(f_dir, minmax_dir):
    df = pd.read_csv(f_dir)
    df["ID"] = pd.to_numeric(df["ID"], errors='coerce')
    df["region"] = pd.to_numeric(df["region"], errors='coerce')
    df_minmax = pd.DataFrame()
    df_minmax['max'] = df.max(axis=0)
    df_minmax['min'] = df.min(axis=0)
    df_minmax = df_minmax.T
    df_minmax["region"] = [1, 672]
    df_minmax["region_big"] = [1, 15]
    df_minmax.to_excel(minmax_dir)



def input(group, csv_dir, minmax_dir):
    df_minmax = pd.read_excel(minmax_dir)
    df_ori_ = pd.read_excel(csv_dir)
    # df_ori_ = pd.read_csv(csv_dir)
    features = eval("feature_" + group)

    # Check whether complex values exist within features_all and make adjustments accordingly.

    processed_data = process_dataframe_complex_columns(df_ori_, ['shape_MajorAxisLength', 'shape_MinorAxisLength',
                                                                 'shape_LeastAxisLength', 'shape_Elongation', 'shape_Flatness'])
    df_raw_ = processed_data[features]


    # df_raw_ = df_ori_[features]  # no complex values

    df_area = pd.read_excel('./cluster_model/672areasvolume.xlsx')
    region_num = []
    region_big = []
    for i in range(df_raw_.shape[0]):
        region = df_raw_.iloc[i]["region"][1:-1]
        r_num = int(df_area[df_area['Region_small'] == region]["num"].values[0])
        r_big = int(df_area[df_area['Region_small'] == region]["num_big"].values[0])
        region_num.append(r_num)
        region_big.append(r_big)
    df_raw_["region"] = region_num
    df_raw_.insert(loc=df_raw_.shape[1], column='region_big', value=region_big)
    # Perform normalisation as required.
    df_raw = df_raw_.iloc[:, 3:]
    df_normalized_data = pd.DataFrame()
    for z in range(df_raw.shape[1]):
        fname = df_raw.columns[z]
        df_raw_z = df_raw.iloc[:, z]
        maxz = df_minmax[fname].iloc[0]
        minz = df_minmax[fname].iloc[1]
        df_normalized_data[fname] = df_raw_z.apply(lambda x: (x - minz) / (maxz - minz))

    return df_raw_["ID"], df_normalized_data


def kmeans(df_raw, df_normalized_data, out_dir):
    kms = KMeans(n_clusters=3, init='k-means++')
    data_fit = kms.fit(df_normalized_data)
    centers = data_fit.cluster_centers_
    labs = data_fit.labels_
    # cls = df_raw["cls"]
    df_labels = DataFrame(kms.labels_)
    df_labels.to_csv(out_dir + 'kms_label.csv')
    df_A_0 = df_normalized_data[labs == 0]
    df_A_1 = df_normalized_data[labs == 1]
    df_A_2 = df_normalized_data[labs == 2]
    # df_A_3 = df_normalized_data[labs == 3]
    m = np.shape(df_A_0)[1]
    df_A_0.insert(df_A_0.shape[1], 'label', 0)
    df_A_1.insert(df_A_1.shape[1], 'label', 1)
    df_A_2.insert(df_A_2.shape[1], 'label', 2)
    # df_A_3.insert(df_A_3.shape[1], 'label', 3)
    df_labels_data = pd.concat([df_A_0, df_A_1, df_A_2])
    df_labels_data.to_csv(out_dir + 'f_kms_labeled.csv')
    df_centers = DataFrame(centers)
    df_centers.to_csv(out_dir + 'kms_center.csv')

    pca = PCA(n_components=2)
    pca.fit(df_normalized_data)
    data_pca = pca.transform(df_normalized_data)
    data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    data_pca.insert(data_pca.shape[1], 'labels', labs)

    pca = PCA(n_components=2)
    pca.fit(centers)
    data_pca_centers = pca.transform(centers)
    data_pca_centers = pd.DataFrame(data_pca_centers, columns=['PC1', 'PC2'])
    # Visualize it:
    plt.figure(figsize=(8, 6))
    # plt.scatter(data_pca.values[:, 0], data_pca.values[:, 1], s=3, c=data_pca.values[:, 2], cmap='Accent')
    # plt.scatter(data_pca_centers.values[:, 0], data_pca_centers.values[:, 1], marker='o', s=55, c='#8E00FF')
    plt.scatter(data_pca.values[:, 0], df_normalized_data.values[:, 0], s=3, c=data_pca.values[:, 2], cmap='Accent')
    plt.scatter(data_pca_centers.values[:, 0], df_centers.values[:, 0], marker='o', s=55, c='#8E00FF')
    plt.show()
    os.makedirs(out_dir + 'figs', exist_ok=True)
    plt.savefig(out_dir + 'figs/kms-3.png')
    # return df_labels_data, df_centers


def InitialCentroid(x, K):
    c0_idx = int(np.random.uniform(0, len(x)))
    centroid = x[c0_idx].reshape(1, -1)
    k = 1
    n = x.shape[0]
    while k < K:
        d2 = []
        for i in range(n):
            subs = centroid - x[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            d2.append(np.min(dimension_s))
        # new_c_idx = np.argmax(d2)
        prob = np.array(d2) / np.sum(np.array(d2))
        new_c_idx = np.random.choice(n, p=prob)
        centroid = np.vstack([centroid, x[new_c_idx]])
        k += 1
    return centroid


def kmeans_plus(k, df_raw, df_normalized_data, out_dir):
    # cls = df_raw["cls"]
    # K = len(np.unique(cls))
    kms_plus = KMeans(n_clusters=k, init='k-means++')
    kms_plus.fit(df_normalized_data)

    labels_num = pd.Series(kms_plus.labels_).value_counts()
    labels = pd.DataFrame(kms_plus.labels_)
    centers = pd.DataFrame(kms_plus.cluster_centers_)
    r = pd.concat([centers, labels_num], axis=1)  #
    r.columns = list(df_raw.columns) + [u'label_num']
    r.to_csv(out_dir + 'kmsp_labelnum-' + str(k) + '.csv')

    r1 = pd.concat([df_raw, pd.Series(kms_plus.labels_, index=df_raw.index)], axis=1)
    r1.columns = list(df_raw.columns) + [u'label']
    r1.to_csv(out_dir + 'f_kmsp_labeled-' + str(k) + '.csv')

    r2 = pd.Series(kms_plus.labels_, index=df_raw.index)
    r2.columns = [u'label']
    r2.to_csv(out_dir + 'kmsp_labeled-' + str(k) + '.csv')

    pickle.dump(kms_plus, open(out_dir + "kmp-" + str(k) + ".pkl", "wb"))

    return labels, centers, r1



def predict(model, data, n, out_dir, ID):
    # kmp_model = pickle.load(open(model, 'rb'))
    kmp_model = joblib.load(filename=model)
    data = np.array(data)
    np.nan_to_num(data, copy=False)
    y_pred = kmp_model.predict(data)
    y_pred_1 = [round(value) for value in y_pred]
    # y_pred = kmp_model.predict(data)
    # y_pred_1 = [round(value) for value in y_pred]
    cls = pd.DataFrame(y_pred)
    cls.columns = ["cls"]
    cls["ID"] = ID
    # df_raw_.insert(loc=df_raw_.shape[1], column='region_big', value=region_big)
    cls.to_csv(out_dir + "cls_" + str(n) + ".csv")
    cls_1 = pd.DataFrame(y_pred_1)
    cls_1.columns = ["cls"]
    cls_1["ID"] = ID
    cls_1.to_csv(out_dir + "round_cls_" + str(n) + ".csv")



if __name__ == '__main__':
    base_dir = './cluster_test/'
    model_dir = './cluster_model/'
    f_dir = base_dir + "test.xlsx"


    '''Clustering model training'''
    # # Determine the coefficient and select the most appropriate number of clusters.
    # raw_data = pd.read_excel(f_dir)[feature_multi]   # Feature groups shall be selected as required.
    # normalized_data = raw_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # out_dir = base_dir + "retrain/"
    # if "retrain" not in os.listdir(base_dir):
    #     os.mkdir(out_dir)
    # # Four methods for selecting cluster numbers.
    # shouzhou(normalized_data, out_dir)
    # K_, max_ = contour(normalized_data, out_dir)
    # K_CH, max_CH = CH(normalized_data, out_dir)
    # K_DB, max_DB = DB(normalized_data, out_dir)
    # # kmeans++: Perform clustering using the selected number of clusters.
    # labels_, centers_, r_ = kmeans_plus(21, raw_data, normalized_data, out_dir)
    # sne_show(normalized_data, r_, 21, out_dir)


    '''Subtype prediction'''
    # Models
    model_gray = model_dir + "gray_kmp-2.pkl"
    model_shape = model_dir + "shape_kmp-5.pkl"
    model_texture = model_dir + "texture_kmp-8.pkl"
    model_multi = model_dir + "multi_kmp-21.pkl"

    for group in ["gray", "shape", "texture", "multi"]:
        minmax_dir = model_dir + group + "_f_minmax.xlsx"

        out_dir = base_dir + group + "/"
        if group not in os.listdir(base_dir):
            os.mkdir(out_dir)

        ID_, normalized_data = input(group, f_dir, minmax_dir)
        model = eval("model_" + group)
        len = normalized_data.shape[0]  # When the number of predictions is too large, they must be predicted in batches.

        if len>1000:
            num = round(len / 1000)
            for n in range(num):
                if n != num - 1:
                    part_data = normalized_data.iloc[(1000 * n):(1000 * (n + 1)), :]
                    ID = ID_.iloc[(1000 * n):(1000 * (n + 1))].reset_index(drop=True)
                else:
                    part_data = normalized_data.iloc[(1000 * n):, :]
                    ID = ID_.iloc[(1000 * n):].reset_index(drop=True)
                predict(model, part_data, n, out_dir, ID)
            # Results combined.
            for i in range(num):
                cls_path = out_dir + "cls_" + str(i) + ".csv"
                round_path = out_dir + "round_cls_" + str(i) + ".csv"
                df_cls = pd.read_csv(cls_path, header=0)
                df_round = pd.read_csv(round_path, header=0)
                if i == 0:
                    df_cls.to_csv(out_dir + 'unite_5xcls.csv', encoding="utf-8", index=False, header=True, mode='a+')
                    df_round.to_csv(out_dir + 'unite_5xcls_round.csv', encoding="utf-8", index=False, header=True,
                                    mode='a+')
                else:
                    df_cls.to_csv(out_dir + 'unite_5xcls.csv', encoding="utf-8", index=False, header=False, mode='a+')
                    df_round.to_csv(out_dir + 'unite_5xcls_round.csv', encoding="utf-8", index=False, header=False,
                                    mode='a+')
        else:
            predict(model, normalized_data, 0, out_dir, ID_)
