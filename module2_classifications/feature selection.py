'''

Feature selection

Used for screening morphotype-related characteristics.
The components provided are solely for analytical processing;
each component requires manual post-processing, such as tabulation,
manual screening, and integration to obtain the final screening results.

Components included:
1. Wasserstein-distance,
2. Jensen-Shannon Divergence,
3. Kruskal-Wallis H Test,
4. Principal Component Analysis,
5. Kernel Density Estimation.

'''


from scipy.stats import wasserstein_distance as wd
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


cls_list = {1:"Diffuse", 2:"Core", 3:"CAA"}


def w_distance(csv, output_dir):
    index = csv.columns
    w_output = pd.DataFrame()
    for fn in feature_names:
        # fn_ = "original_" + fn
        fn_ = fn
        cls1 = []
        cls2 = []
        cls3 = []
        if fn_ not in index:
            continue
        else:
            for i in range(csv.shape[0]):
                value = csv[fn_][i]
                cls = csv["cls"][i]
                if cls == 1:
                    cls1.append(value)
                elif cls == 2:
                    cls2.append(value)
                elif cls == 3:
                    cls3.append(value)

        for i in range(5):
            cls_other = []
            if i == 1:
                cls_other.extend(cls2)
                cls_other.extend(cls3)
                w1_2 = wd(cls1, cls2)
                w1_3 = wd(cls1, cls3)
                w1_min = min(w1_2, w1_3)
                w1_other = wd(cls1, cls_other)
            elif i == 2:
                cls_other.extend(cls1)
                cls_other.extend(cls3)
                w2_1 = wd(cls2, cls1)
                w2_3 = wd(cls2, cls3)
                w2_min = min(w2_1, w2_3)
                w2_other = wd(cls2, cls_other)
            elif i == 3:
                cls_other.extend(cls1)
                cls_other.extend(cls2)
                w3_1 = wd(cls3, cls1)
                w3_2 = wd(cls3, cls2)
                w3_min = min(w3_1, w3_2)
                w3_other = wd(cls3, cls_other)

        w_output[fn_] = pd.Series({"w1_2": w1_2,
                                   "w1_3": w1_3,
                                   "w1_min": w1_min,
                                   "w1_other": w1_other,
                                   "w2_1": w2_1,
                                   "w2_3": w2_3,
                                   "w2_min": w2_min,
                                   "w2_other": w2_other,
                                   "w3_1": w3_1,
                                   "w3_2": w3_2,
                                   "w3_min": w3_min,
                                   "w3_other": w3_other})
    w_output.to_excel(output_dir)


def js_divergence(p, q):  # 0 for identical, 1 for opposite
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def js_div(arr1, arr2, num_bins):  # Divide the data into equally-spaced intervals (bins), then use the probability of samples falling into each bin as the distribution, num_bins
    max0 = max(np.max(arr1), np.max(arr2))
    min0 = min(np.min(arr1), np.min(arr2))
    bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)  # Generate a uniform distribution
    PDF1 = pd.cut(arr1, bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2, bins).value_counts() / len(arr2)
    return js_divergence(PDF1.values, PDF2.values)


def js_calcute(csv, output_dir):
    index = csv.columns
    js_output = pd.DataFrame()
    for fn in feature_names:
        # fn_ = "original_" + fn
        fn_ = fn
        cls1 = []
        cls2 = []
        cls3 = []
        if fn_ not in index:
            continue
        else:
            for i in range(csv.shape[0]):
                value = csv[fn_][i]
                cls = csv["cls"][i]
                if cls == 1:
                    cls1.append(value)
                elif cls == 2:
                    cls2.append(value)
                elif cls == 3:
                    cls3.append(value)

        for i in range(5):
            cls_other = []
            if i == 1:
                cls_other.extend(cls2)
                cls_other.extend(cls3)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                js1_2 = js_div(cls1, cls2, num_bins=58)
                js1_3 = js_div(cls1, cls3, num_bins=58)
                js1_min = min(js1_2, js1_3)
                js1_other = js_div(cls1, cls_other, num_bins=372)
            elif i == 2:
                cls_other.extend(cls1)
                cls_other.extend(cls3)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                js2_1 = js_div(cls2, cls1, num_bins=58)
                js2_3 = js_div(cls2, cls3, num_bins=32)
                js2_min = min(js2_1, js2_3)
                js2_other = js_div(cls2, cls_other, num_bins=372)
            elif i == 3:
                cls_other.extend(cls1)
                cls_other.extend(cls2)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                js3_1 = js_div(cls3, cls1, num_bins=58)
                js3_2 = js_div(cls3, cls2, num_bins=32)
                js3_min = min(js3_1, js3_2)
                js3_other = js_div(cls3, cls_other, num_bins=372)

        js_output[fn_] = pd.Series({"js1_2": js1_2,
                                    "js1_3": js1_3,
                                    "js1_min": js1_min,
                                    "js1_other": js1_other,
                                    "js2_1": js2_1,
                                    "js2_3": js2_3,
                                    "js2_min": js2_min,
                                    "js2_other": js2_other,
                                    "js3_1": js3_1,
                                    "js3_2": js3_2,
                                    "js3_min": js3_min,
                                    "js3_other": js3_other})

    js_output.to_excel(output_dir)


def kw_test(csv, output_dir):
    index = csv.columns
    kw_output = pd.DataFrame()
    for fn in feature_names:
        # fn_ = "original_" + fn
        fn_ = fn
        cls1 = []
        cls2 = []
        cls3 = []
        if fn_ not in index:
            continue
        else:
            for i in range(csv.shape[0]):
                value = csv[fn_][i]
                cls = csv["cls"][i]
                if cls == 1:
                    cls1.append(value)
                elif cls == 2:
                    cls2.append(value)
                elif cls == 3:
                    cls3.append(value)

        for i in range(5):
            cls_other = []
            if i == 1:
                cls_other.extend(cls2)
                cls_other.extend(cls3)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                kw1_2, p1_2 = scipy.stats.kruskal(cls1, cls2)
                kw1_3, p1_3 = scipy.stats.kruskal(cls1, cls3)
                p1_min = min(p1_2, p1_3)
                kw1_other, p1_other = scipy.stats.kruskal(cls1, cls_other)
            elif i == 2:
                cls_other.extend(cls1)
                cls_other.extend(cls3)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                kw2_1, p2_1 = scipy.stats.kruskal(cls2, cls1)
                kw2_3, p2_3 = scipy.stats.kruskal(cls2, cls3)
                p2_min = min(p2_1, p2_3)
                kw2_other, p2_other = scipy.stats.kruskal(cls2, cls_other)
            elif i == 3:
                cls_other.extend(cls1)
                cls_other.extend(cls2)
                cls1 = np.asarray(cls1)
                cls2 = np.asarray(cls2)
                cls3 = np.asarray(cls3)
                cls_other = np.asarray(cls_other)
                kw3_1, p3_1 = scipy.stats.kruskal(cls3, cls1)
                kw3_2, p3_2 = scipy.stats.kruskal(cls3, cls2)
                p3_min = min(p3_1, p3_2)
                kw3_other, p3_other = scipy.stats.kruskal(cls3, cls_other)

        kw_output[fn_] = pd.Series({"kw_p-1_2": p1_2,
                                    "kw_p-1_3": p1_3,
                                    "kw_p-1_min": p1_min,
                                    "kw_p-1_other": p1_other,
                                    "kw_p-2_1": p2_1,
                                    "kw_p-2_3": p2_3,
                                    "kw_p-2_min": p2_min,
                                    "kw_p-2_other": p2_other,
                                    "kw_p-3_1": p3_1,
                                    "kw_p-3_2": p3_2,
                                    "kw_p-3_min": p3_min,
                                    "kw_p-3_other": p3_other})

    kw_output.to_excel(output_dir)



def PCA_feature(df, feature_names, output_dir):
    with pd.ExcelWriter(output_dir, engine='openpyxl') as writer:
        for cls in [1, 2, 3]:
            cls_name = cls_list[cls]
            X = df[df["cls"] == cls][feature_names].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            loadings = pca.components_
            # Composite Importance (using the top three principal components)
            importance_scores = np.sum(
                np.abs(loadings[:3]) * pca.explained_variance_ratio_[:3].reshape(-1, 1),
                axis=0
            )

            results = pd.DataFrame({
                'Features': feature_names,
                'Composite Importance': importance_scores,
                'PC1': loadings[0],
                'PC2': loadings[1],
                'PC3': loadings[2]
            })
            results = results.sort_values('Composite Importance', ascending=False)
            results['Ranks'] = range(1, len(results) + 1)
            results = results[['Ranks', 'Features', 'Composite Importance', 'PC1', 'PC2', 'PC3']]
            results.to_excel(writer, sheet_name=cls_name, index=False)



def KDE_plot(df, feature_names, output_dir):
    for feature in feature_names:
        plt.figure(figsize=(10, 10))
        # Histogram
        sns.kdeplot(data=df[df["cls"] == 1][feature], label="diffuse", color='royalblue', shade=True, linewidth=2)
        sns.kdeplot(data=df[df["cls"] == 2][feature], label="core", color='g', shade=True, linewidth=2)
        sns.kdeplot(data=df[df["cls"] == 3][feature], label="caa", color='r', shade=True, linewidth=2)
        # sns.kdeplot(current_palette)
        ax = plt.gca()
        # ax.set_xlim(f_min, f_max)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        plt.tick_params(width=1, length=10, pad=5)
        # plt.legend()
        plt.savefig(output_dir + feature + ".png", dpi=300)   #transparent=True,
        plt.close()



if __name__ == '__main__':
    csv_dir = "./test/test.xlsx"
    csv = pd.read_excel(csv_dir)
    feature_names = csv.columns

    wd_dir = "./test/test_wd.xlsx"
    w_distance(csv, wd_dir)

    js_dir = "./test/test_js.xlsx"
    js_calcute(csv, js_dir)

    kw_dir = "./test/test_kw.xlsx"
    kw_test(csv, kw_dir)

    pca_dir = "./test/test_pca.xlsx"
    PCA_feature(csv, feature_names, pca_dir)

    kde_dir = "./test/KDE_plot/"
    if "KDE_plot" not in os.listdir("./test/"):
        os.mkdir(kde_dir)
    KDE_plot(csv, feature_names, kde_dir)

