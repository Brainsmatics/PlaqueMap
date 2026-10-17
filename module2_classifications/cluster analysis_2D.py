'''

Cluster analysis - 2D version

Comprising two parts: model training and prediction.
During training, a comprehensive selection can be made through four cluster number selection methods.

An example of the operation can be found in the main function.
Comprising four feature groups: Gray, Shape, Texture, Multi.

We perform feature mapping from 3D mouse data to 2D before retraining the model.
Users may omit feature mapping if their data is already in 2D format.

'''



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist   # Compute the pairwise distance between two input collections
from pandas import DataFrame
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pickle
import joblib
import os
import numpy as np



shape3Dto2D = {"shape_VoxelVolume":'shape2D_PixelSurface',
               "shape_MeshVolume":'shape2D_MeshSurface',
               "shape_SurfaceArea":'shape2D_Perimeter',
               "shape_SurfaceVolumeRatio":'shape2D_PerimeterSurfaceRatio',
               "shape_Sphericity":'shape2D_Sphericity',
               "shape_SphericalDisproportion":'shape2D_SphericalDisproportion',
               "shape_Maximum3DDiameter":'shape2D_MaximumDiameter',
               "shape_MajorAxisLength":'shape2D_MajorAxisLength',
               "shape_MinorAxisLength":'shape2D_MinorAxisLength',
               "shape_Elongation":'shape2D_Elongation',
               "volume_regist": "volume"}
feature_names_2D = ['shape2D_MeshSurface',
                     'shape2D_PixelSurface',
                     'shape2D_Perimeter',
                     'shape2D_PerimeterSurfaceRatio',
                     'shape2D_Sphericity',
                     'shape2D_SphericalDisproportion',
                     'shape2D_MaximumDiameter',
                     'shape2D_MajorAxisLength',
                     'shape2D_MinorAxisLength',
                     'shape2D_Elongation',
                     'glcm_Autocorrelation',
                     'glcm_JointAverage',
                     'glcm_ClusterProminence',
                     'glcm_ClusterShade',
                     'glcm_ClusterTendency',
                     'glcm_Contrast',
                     'glcm_Correlation',
                     'glcm_DifferenceAverage',
                     'glcm_DifferenceEntropy',
                     'glcm_DifferenceVariance',
                     'glcm_JointEnergy',
                     'glcm_JointEntropy',
                     'glcm_Imc1',
                     'glcm_Imc2',
                     'glcm_Idm',
                     'glcm_Idmn',
                     'glcm_Id',
                     'glcm_Idn',
                     'glcm_InverseVariance',
                     'glcm_MaximumProbability',
                     'glcm_SumEntropy',
                     'glcm_SumSquares',
                     'glszm_GrayLevelVariance',
                     'glszm_ZoneVariance',
                     'glszm_SizeZoneNonUniformity',
                     'glszm_GrayLevelNonUniformity',
                     'glszm_SmallAreaHighGrayLevelEmphasis',
                     'glszm_LargeAreaLowGrayLevelEmphasis',
                     'glszm_LargeAreaHighGrayLevelEmphasis',
                     'glszm_SmallAreaLowGrayLevelEmphasis',
                     'glszm_SmallAreaEmphasis',
                     'glszm_LargeAreaEmphasis',
                     'glszm_GrayLevelNonUniformityNormalized',
                     'glszm_SizeZoneNonUniformityNormalized',
                     'glszm_ZonePercentage',
                     'glszm_ZoneEntropy',
                     'glszm_LowGrayLevelZoneEmphasis',
                     'glszm_HighGrayLevelZoneEmphasis',
                     'ngtdm_Coarseness',
                     'ngtdm_Contrast',
                     'ngtdm_Busyness',
                     'ngtdm_Complexity',
                     'ngtdm_Strength',
                     'gldm_SmallDependenceEmphasis',
                     'gldm_LargeDependenceEmphasis',
                     'gldm_GrayLevelNonUniformity',
                     'gldm_DependenceNonUniformity',
                     'gldm_DependenceNonUniformityNormalized',
                     'gldm_GrayLevelVariance',
                     'gldm_DependenceVariance',
                     'gldm_DependenceEntropy',
                     'gldm_LowGrayLevelEmphasis',
                     'gldm_HighGrayLevelEmphasis',
                     'gldm_SmallDependenceLowGrayLevelEmphasis',
                     'gldm_SmallDependenceHighGrayLevelEmphasis',
                     'gldm_LargeDependenceLowGrayLevelEmphasis',
                     'gldm_LargeDependenceHighGrayLevelEmphasis',
                     "r","max_gray","value_min","ratio","volume"]
# pick
feature_multi = ["r", "max_gray","value_min","ratio","volume","shape_VoxelVolume","shape_MeshVolume","shape_SurfaceArea",
               "shape_SurfaceVolumeRatio","shape_Compactness1","shape_Compactness2","shape_Sphericity","shape_SphericalDisproportion",
               "shape_Maximum3DDiameter","shape_Maximum2DDiameterSlice","shape_Maximum2DDiameterColumn",
               "shape_Maximum2DDiameterRow","shape_MajorAxisLength","shape_MinorAxisLength","shape_LeastAxisLength",
               "shape_Elongation","shape_Flatness","glcm_Autocorrelation","glcm_JointAverage","glcm_ClusterProminence",
               "glcm_ClusterShade","glcm_ClusterTendency","glcm_Contrast","glcm_Correlation","glcm_DifferenceAverage",
               "glcm_DifferenceEntropy","glcm_DifferenceVariance","glcm_JointEnergy","glcm_JointEntropy","glcm_Imc1",
               "glcm_Imc2","glcm_Idm","glcm_Idmn","glcm_Id","glcm_Idn","glcm_InverseVariance","glcm_MaximumProbability",
               "glcm_SumEntropy","glcm_SumSquares","glszm_GrayLevelVariance","glszm_ZoneVariance","glszm_SizeZoneNonUniformity",
               "glszm_GrayLevelNonUniformity","glszm_SmallAreaHighGrayLevelEmphasis","glszm_LargeAreaLowGrayLevelEmphasis",
               "glszm_LargeAreaHighGrayLevelEmphasis","glszm_SmallAreaLowGrayLevelEmphasis","glszm_SmallAreaEmphasis",
               "glszm_LargeAreaEmphasis","glszm_GrayLevelNonUniformityNormalized","glszm_SizeZoneNonUniformityNormalized",
               "glszm_ZonePercentage","glszm_ZoneEntropy","glszm_LowGrayLevelZoneEmphasis","glszm_HighGrayLevelZoneEmphasis",
               "ngtdm_Coarseness","ngtdm_Contrast","ngtdm_Busyness","ngtdm_Complexity","ngtdm_Strength","gldm_SmallDependenceEmphasis",
               "gldm_LargeDependenceEmphasis","gldm_GrayLevelNonUniformity","gldm_DependenceNonUniformity",
               "gldm_DependenceNonUniformityNormalized","gldm_GrayLevelVariance","gldm_DependenceVariance","gldm_DependenceEntropy",
               "gldm_LowGrayLevelEmphasis","gldm_HighGrayLevelEmphasis","gldm_SmallDependenceLowGrayLevelEmphasis",
               "gldm_SmallDependenceHighGrayLevelEmphasis","gldm_LargeDependenceLowGrayLevelEmphasis",
               "gldm_LargeDependenceHighGrayLevelEmphasis","region"]
feature_shape = ["r","value_min","ratio","volume","shape_VoxelVolume","shape_MeshVolume","shape_SurfaceArea",
                 "shape_SurfaceVolumeRatio","shape_Compactness1","shape_Compactness2","shape_Sphericity","shape_SphericalDisproportion",
                 "shape_Maximum3DDiameter","shape_Maximum2DDiameterSlice","shape_Maximum2DDiameterColumn",
                 "shape_Maximum2DDiameterRow","shape_MajorAxisLength","shape_MinorAxisLength","shape_LeastAxisLength",
                 "shape_Elongation","shape_Flatness","glszm_ZoneVariance","glszm_SizeZoneNonUniformity",
                 "glszm_SizeZoneNonUniformityNormalized","region"]
feature_gray = ["max_gray","glcm_JointAverage","glcm_ClusterProminence","glcm_ClusterShade","glcm_ClusterTendency",
                "glcm_Contrast","glcm_Correlation","glcm_DifferenceAverage",
                "glcm_DifferenceEntropy","glcm_DifferenceVariance","glcm_JointEnergy",       #"glcm_JointEntropy"(this feature is missing from the gray model)
                "glcm_Idm","glcm_Idmn","glcm_Id","glcm_Idn",
                "glcm_InverseVariance","glcm_MaximumProbability","glcm_SumEntropy","glcm_SumSquares",
                "glszm_GrayLevelVariance","glszm_GrayLevelNonUniformity","glszm_SmallAreaHighGrayLevelEmphasis",
                "glszm_LargeAreaLowGrayLevelEmphasis","glszm_LargeAreaHighGrayLevelEmphasis","glszm_SmallAreaLowGrayLevelEmphasis",
                "glszm_GrayLevelNonUniformityNormalized","glszm_LowGrayLevelZoneEmphasis","glszm_HighGrayLevelZoneEmphasis",
                "ngtdm_Contrast","ngtdm_Busyness","ngtdm_Complexity","ngtdm_Strength",
                "gldm_GrayLevelNonUniformity","gldm_DependenceNonUniformity",
                "gldm_DependenceNonUniformityNormalized","gldm_GrayLevelVariance","gldm_DependenceVariance","gldm_DependenceEntropy",
                "gldm_LowGrayLevelEmphasis","gldm_HighGrayLevelEmphasis","gldm_SmallDependenceLowGrayLevelEmphasis",
                "gldm_SmallDependenceHighGrayLevelEmphasis","gldm_LargeDependenceLowGrayLevelEmphasis",
                "gldm_LargeDependenceHighGrayLevelEmphasis","region"]
feature_texture = ["glcm_Autocorrelation","glcm_Imc1","glcm_Imc2","glszm_SmallAreaEmphasis",
                   "glszm_LargeAreaEmphasis","glszm_ZonePercentage","glszm_ZoneEntropy","ngtdm_Coarseness",
                   "gldm_SmallDependenceEmphasis","gldm_LargeDependenceEmphasis","region"]

feature_group = {"feature_multi":["r", "max_gray","value_min","ratio","volume","shape_VoxelVolume","shape_MeshVolume","shape_SurfaceArea",
               "shape_SurfaceVolumeRatio","shape_Compactness1","shape_Compactness2","shape_Sphericity","shape_SphericalDisproportion",
               "shape_Maximum3DDiameter","shape_Maximum2DDiameterSlice","shape_Maximum2DDiameterColumn",
               "shape_Maximum2DDiameterRow","shape_MajorAxisLength","shape_MinorAxisLength","shape_LeastAxisLength",
               "shape_Elongation","shape_Flatness","glcm_Autocorrelation","glcm_JointAverage","glcm_ClusterProminence",
               "glcm_ClusterShade","glcm_ClusterTendency","glcm_Contrast","glcm_Correlation","glcm_DifferenceAverage",
               "glcm_DifferenceEntropy","glcm_DifferenceVariance","glcm_JointEnergy","glcm_JointEntropy","glcm_Imc1",
               "glcm_Imc2","glcm_Idm","glcm_Idmn","glcm_Id","glcm_Idn","glcm_InverseVariance","glcm_MaximumProbability",
               "glcm_SumEntropy","glcm_SumSquares","glszm_GrayLevelVariance","glszm_ZoneVariance","glszm_SizeZoneNonUniformity",
               "glszm_GrayLevelNonUniformity","glszm_SmallAreaHighGrayLevelEmphasis","glszm_LargeAreaLowGrayLevelEmphasis",
               "glszm_LargeAreaHighGrayLevelEmphasis","glszm_SmallAreaLowGrayLevelEmphasis","glszm_SmallAreaEmphasis",
               "glszm_LargeAreaEmphasis","glszm_GrayLevelNonUniformityNormalized","glszm_SizeZoneNonUniformityNormalized",
               "glszm_ZonePercentage","glszm_ZoneEntropy","glszm_LowGrayLevelZoneEmphasis","glszm_HighGrayLevelZoneEmphasis",
               "ngtdm_Coarseness","ngtdm_Contrast","ngtdm_Busyness","ngtdm_Complexity","ngtdm_Strength","gldm_SmallDependenceEmphasis",
               "gldm_LargeDependenceEmphasis","gldm_GrayLevelNonUniformity","gldm_DependenceNonUniformity",
               "gldm_DependenceNonUniformityNormalized","gldm_GrayLevelVariance","gldm_DependenceVariance","gldm_DependenceEntropy",
               "gldm_LowGrayLevelEmphasis","gldm_HighGrayLevelEmphasis","gldm_SmallDependenceLowGrayLevelEmphasis",
               "gldm_SmallDependenceHighGrayLevelEmphasis","gldm_LargeDependenceLowGrayLevelEmphasis",
               "gldm_LargeDependenceHighGrayLevelEmphasis","region"], "feature_shape":["r","value_min","ratio","volume","shape_VoxelVolume","shape_MeshVolume","shape_SurfaceArea",
                 "shape_SurfaceVolumeRatio","shape_Compactness1","shape_Compactness2","shape_Sphericity","shape_SphericalDisproportion",
                 "shape_Maximum3DDiameter","shape_Maximum2DDiameterSlice","shape_Maximum2DDiameterColumn",
                 "shape_Maximum2DDiameterRow","shape_MajorAxisLength","shape_MinorAxisLength","shape_LeastAxisLength",
                 "shape_Elongation","shape_Flatness","glszm_ZoneVariance","glszm_SizeZoneNonUniformity",
                 "glszm_SizeZoneNonUniformityNormalized","region"], "feature_gray":["max_gray","glcm_JointAverage","glcm_ClusterProminence","glcm_ClusterShade","glcm_ClusterTendency",
                "glcm_Contrast","glcm_Correlation","glcm_DifferenceAverage",
                "glcm_DifferenceEntropy","glcm_DifferenceVariance","glcm_JointEnergy",       #"glcm_JointEntropy"(this feature is missing from the gray model)
                "glcm_Idm","glcm_Idmn","glcm_Id","glcm_Idn",
                "glcm_InverseVariance","glcm_MaximumProbability","glcm_SumEntropy","glcm_SumSquares",
                "glszm_GrayLevelVariance","glszm_GrayLevelNonUniformity","glszm_SmallAreaHighGrayLevelEmphasis",
                "glszm_LargeAreaLowGrayLevelEmphasis","glszm_LargeAreaHighGrayLevelEmphasis","glszm_SmallAreaLowGrayLevelEmphasis",
                "glszm_GrayLevelNonUniformityNormalized","glszm_LowGrayLevelZoneEmphasis","glszm_HighGrayLevelZoneEmphasis",
                "ngtdm_Contrast","ngtdm_Busyness","ngtdm_Complexity","ngtdm_Strength",
                "gldm_GrayLevelNonUniformity","gldm_DependenceNonUniformity",
                "gldm_DependenceNonUniformityNormalized","gldm_GrayLevelVariance","gldm_DependenceVariance","gldm_DependenceEntropy",
                "gldm_LowGrayLevelEmphasis","gldm_HighGrayLevelEmphasis","gldm_SmallDependenceLowGrayLevelEmphasis",
                "gldm_SmallDependenceHighGrayLevelEmphasis","gldm_LargeDependenceLowGrayLevelEmphasis",
                "gldm_LargeDependenceHighGrayLevelEmphasis","region"], "feature_texture":["glcm_Autocorrelation","glcm_Imc1","glcm_Imc2","glszm_SmallAreaEmphasis",
                "glszm_LargeAreaEmphasis","glszm_ZonePercentage","glszm_ZoneEntropy","ngtdm_Coarseness",
                "gldm_SmallDependenceEmphasis","gldm_LargeDependenceEmphasis","region"]}



cls_num = {"multi":21, "gray":2, "shape":5, "texture":8}


def kmeans(df_raw, df_normalized_data, out_dir):
    kms = KMeans(n_clusters=3, init='k-means++')
    data_fit = kms.fit(df_normalized_data)
    centers = data_fit.cluster_centers_
    labs = data_fit.labels_
    # cls = df_raw["cls"]
    df_labels = DataFrame(kms.labels_)
    df_labels.to_csv(out_dir + 'kms_label-3.csv')
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
    df_labels_data.to_csv(out_dir + 'f_kms_labeled-3.csv')
    df_centers = DataFrame(centers)
    df_centers.to_csv(out_dir + 'kms_center-3.csv')

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


def kmeans_plus(k, df_raw, df_normalized_data, group, out_dir):
    # cls = df_raw["cls"]
    # K = len(np.unique(cls))
    kms_plus = KMeans(n_clusters=k, init='k-means++')
    kms_plus.fit(df_normalized_data)

    labels_num = pd.Series(kms_plus.labels_).value_counts()
    labels = pd.DataFrame(kms_plus.labels_)
    centers = pd.DataFrame(kms_plus.cluster_centers_)
    r = pd.concat([centers, labels_num], axis=1)
    print(r)
    r.columns = list(df_raw.columns) + [u'label_num']
    os.makedirs(out_dir + '/' + group, exist_ok=True)
    r.to_csv(out_dir + '/'+ group + '/kmsp_labelnum-' + str(k) + '.csv')

    r1 = pd.concat([df_raw, pd.Series(kms_plus.labels_, index=df_raw.index)], axis=1)
    r1.columns = list(df_raw.columns) + [u'label']
    r1.to_csv(out_dir + '/'+ group + '/f_kmsp_labeled-' + str(k) + '.csv')

    r2 = pd.Series(kms_plus.labels_, index=df_raw.index)
    r2.columns = [u'label']
    r2.to_csv(out_dir + '/'+ group + '/kmsp_labeled-' + str(k) + '.csv')

    pickle.dump(kms_plus, open(out_dir + '/'+ group + "/kmp-" + str(k) + ".pkl", "wb"))

    return labels, centers, r1


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



def sne_show(data, r, k, group, out_dir):
    tsne = TSNE()
    tsne.fit_transform(data)
    tsne = pd.DataFrame(tsne.embedding_, index=data.index)

    plt.rcParams['axes.unicode_minus'] = False

    colors = ['OrangeRed', 'Coral', 'LightSalmon', 'SandyBrown', 'DarkOrange', 'Goldenrod',
              'Gold', 'Yellow', 'Olive', 'DarkGreen', 'ForestGreen', 'LimeGreen',
              'LightGreen', 'Turquoise', 'Teal', 'DarkSlateGray', 'CadetBlue', 'SteelBlue',
              'Navy', 'DarkSlateBlue', 'MediumPurple']
    colors_ = ['DarkGreen', 'SteelBlue', 'OrangeRed']
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
    os.makedirs(out_dir + 'figs/' + group, exist_ok=True)
    plt.savefig(out_dir + 'figs/'+ group + '/sne-' + str(k) + '.png', dpi=600, transparent=True)
    d_all.to_csv(out_dir + 'figs/'+ group + '/sne-d.csv')
    # plt.show()


def sne_show_region(data, r, k, e_e, r_s, t, out_dir):
    tsne = TSNE(early_exaggeration=e_e, random_state=r_s)
    tsne.fit_transform(data)
    tsne = pd.DataFrame(tsne.embedding_, index=data.index)

    plt.rcParams['font.sans-serif'] = ['SimHei']
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

    for region in range(1, 16):
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
        plt.savefig(out_dir + 'figs/region' + str(region) + "-" + str(k) + '-' + str(t) + '.png', transparent=True,
                    dpi=300)
        d_all.to_csv(out_dir + 'figs/region' + str(region) + '.csv')
        plt.close()

    # Plot by class (cls)
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
    cls.to_csv(out_dir + "/cls_" + str(n) + ".csv")
    cls_1 = pd.DataFrame(y_pred_1)
    cls_1.columns = ["cls"]
    cls_1["ID"] = ID
    cls_1.to_csv(out_dir + "/round_cls_" + str(n) + ".csv")



if __name__ == '__main__':
    base_dir = './cluster_test_2D/'
    model_dir = './cluster_model_2D/'
    csv_dir = base_dir + "test_2D.xlsx"

    # raw_data = pd.read_csv(csv_dir)
    # raw_data.rename(columns=shape3Dto2D, inplace=True)
    # normalized_data_s0 = raw_data.iloc[:, :3]
    # normalized_data_s1 = raw_data.iloc[:, 3:].apply(
    #     lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # normalized_data = pd.concat([normalized_data_s0, normalized_data_s1], axis=1)

    normalized_data = pd.read_excel(csv_dir)
    for group in ["shape", "multi", "texture", "gray"]:
        out_dir = base_dir + group + "/"
        if group not in os.listdir(base_dir):
            os.mkdir(out_dir)

        features_ = feature_group[("feature_" + group)]
        features = [shape3Dto2D.get(item, item) for item in features_]
        result = [item for item in features if item in feature_names_2D]

        cls_num_ = cls_num[group]
        ID_ = normalized_data["ID"]
        # all_3D_ = normalized_data["all"]
        normalized_data_ = normalized_data[result]

        ''' Model training '''
        # raw_data_ = raw_data[result]
        # labels__, centers__, r__ = kmeans_plus(cls_num_, raw_data_, normalized_data_, group, base_dir)
        # sne_show(normalized_data_, r__, cls_num_, group, base_dir)

        ''' Model prediction '''
        model = model_dir + group + "_kmp-" + str(cls_num_) + ".pkl"
        len = normalized_data_.shape[0]
        if len > 1000:
            num = round(len / 1000)
            for n in range(num):
                if n != num - 1:
                    part_data = normalized_data_.iloc[(1000 * n):(1000 * (n + 1)), :]
                    ID = ID_.iloc[(1000 * n):(1000 * (n + 1))].reset_index(drop=True)
                    # all_3D = all_3D_.iloc[(1000 * n):(1000 * (n + 1))].reset_index(drop=True)
                else:
                    part_data = normalized_data_.iloc[(1000 * n):, :]
                    ID = ID_.iloc[(1000 * n):].reset_index(drop=True)
                    # all_3D = all_3D_.iloc[(1000 * n):].reset_index(drop=True)
                predict(model, part_data, n, out_dir, ID)
            for i in range(num):
                cls_path = out_dir + "/cls_" + str(i) + ".csv"
                round_path = out_dir + "/round_cls_" + str(i) + ".csv"
                df_cls = pd.read_csv(cls_path, header=0)
                df_round = pd.read_csv(round_path, header=0)
                if i == 0:
                    df_cls.to_csv(out_dir + '/unite_5xcls.csv', encoding="utf-8", index=False, header=True, mode='a+')
                    df_round.to_csv(out_dir + '/unite_5xcls_round.csv', encoding="utf-8", index=False, header=True,
                                    mode='a+')
                else:
                    df_cls.to_csv(out_dir + '/unite_5xcls.csv', encoding="utf-8", index=False, header=False, mode='a+')
                    df_round.to_csv(out_dir + '/unite_5xcls_round.csv', encoding="utf-8", index=False, header=False,
                                    mode='a+')
        else:
            predict(model, normalized_data_, 0, out_dir, ID_)

