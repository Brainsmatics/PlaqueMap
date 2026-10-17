'''

Distribution difference

Includes several metrics:
1. Kolmogorov-Smirnov test (KS test)
    To compare differences in the distribution of subtype numbers across brain regions.
    If the p-value is below the significance threshold, this indicates that the two subtypes exhibit distinct spatial distributions.

2. Multiple tests.
    Employ the test to evaluate differences in feature distribution between subtypes.

    First stage: Single-feature t-tests (conducted individually for each feature).
    Levene's test: Checks whether the variances of two data sets are equal
    Selecting t-tests:
        Homogeneity of variances: Use standard t-test (equal_var=True)
        Heterogeneity of variances: Use Welch's t-test (equal_var=False)

    Second stage: multivariate mean vector test (overall test for all features).
    Test selection criteria:
        Hotelling's T² test: Employed when n1 + n2 > number of features + 1
        Permutation test: Used when sample sizes are insufficient
'''


from scipy.stats import ks_2samp
import scipy.stats as stats
import numpy as np
import pandas as pd
import math
import os

csv_dir = "./distribution/test_data/"                        # input
out_dir = "./distribution/"                                  # output

cls = ["multi", "gray", "shape", "texture"]      # all = multi
cls_num = {"multi":21, "gray":2, "shape":5, "texture":8}
age_group = [1.5, 2, 4, 6, 10]

big = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CB']

# KS test
def cal_differ(csv_dir, out_dir, age):
    if age != 0:
        csv_dir = csv_dir + str(age) + "/"
        out_dir = out_dir + str(age) + "/"
    os.makedirs(out_dir + "out_ks/", exist_ok=True)
    for b in big:
        count_path = csv_dir + b + ".xlsx"
        out_path = out_dir + "out_ks/" + b + "_ks.xlsx"
        writer = pd.ExcelWriter(out_path)

        # Compute the similarity of category count distributions for small regions.
        small = pd.read_excel(count_path, sheet_name="small")
        r_list = small.columns.tolist()
        ss = 0
        out = pd.DataFrame()
        for i in range(len(r_list)-1):
            r1_name = r_list[i + 1]
            r1 = small[r1_name]
            for j in range(len(r_list)-1-i):
                r2_name = r_list[i+1+j]
                r2 = small[r2_name]
                stat_, p_ = ks_2samp(r1, r2)
                if p_ <= 0.05:  # p<0.05, significant difference; p>0.05, no significant difference.
                    differ = 1  # 1: True, there are differences.
                else:
                    differ = 0
                out[ss] = pd.Series({"region1": r1_name,
                                     "region2": r2_name,
                                     "ks_stat": stat_,
                                     "p_value": p_,
                                     "differ": differ})
                ss = ss+1
        out.T.to_excel(writer, sheet_name="small")

        # Compute the similarity of category count distributions for the medium region.
        medium = pd.read_excel(count_path, sheet_name="medium")
        rr_list = medium.columns.tolist()
        mm = 0
        out_m = pd.DataFrame()
        for i in range(len(rr_list) - 1):
            r1_name = rr_list[i + 1]
            r1 = medium[r1_name]
            for j in range(len(rr_list) - 1 - i):
                r2_name = rr_list[i + 1 + j]
                r2 = medium[r2_name]
                stat_, p_ = ks_2samp(r1, r2)
                if p_ <= 0.05:  # p<0.05, significant difference; p>0.05, no significant difference.
                    differ = 1  # 1: True, there are differences.
                else:
                    differ = 0
                out_m[mm] = pd.Series({"region1": r1_name,
                                       "region2": r2_name,
                                       "ks_stat": stat_,
                                       "p_value": p_,
                                       "differ": differ})
                mm = mm + 1
        out_m.T.to_excel(writer, sheet_name="medium")
        writer.close()

cal_differ(csv_dir+"count_regions/", out_dir, 0)
# The distribution of categories across different age groups of mice can be calculated as required.
# for age in age_group:
#     cal_differ(csv_dir, out_dir, age)



# Multiple tests
df = pd.read_csv(csv_dir + "features_test.csv")
df.rename(columns={'all': 'multi'}, inplace=True)
df_nor = df.iloc[:, 3:]
f_list = df_nor.columns.tolist()

out_path = out_dir + "stat_t.xlsx"
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for cls_order in range(4):
        cls_ = cls[cls_order]
        feature_results = []  # Per-feature results
        mean_results = []  # Feature mean comparison results

        num = cls_num[cls_]

        for i in range(num):
            df_1_ = df[df[cls_] == i][f_list]

            # Calculate mean vector for first group
            if df_1_.shape[0] > 0:
                df_1_mean_vector = df_1_.mean(axis=0).values
                n1 = df_1_.shape[0]
            else:
                df_1_mean_vector = np.zeros(len(f_list))
                n1 = 0

            for cls_order_ in range(4 - cls_order):
                cls__ = cls[cls_order_ + cls_order]
                if cls__ == cls_:
                    # Within same subtype
                    for j in range(1, num - i):  # Avoid self-comparison
                        cls2_num = i + j
                        df_2_ = df[df[cls__] == cls2_num][f_list]
                        if df_2_.shape[0] > 0:
                            df_2_mean_vector = df_2_.mean(axis=0).values
                            n2 = df_2_.shape[0]
                        else:
                            df_2_mean_vector = np.zeros(len(f_list))
                            n2 = 0

                        # 1. Per-feature analysis
                        for feature in f_list:
                            # Try to get data for t-test
                            data1 = df_1_[feature].dropna().values if feature in df_1_.columns else np.array([])
                            data2 = df_2_[feature].dropna().values if feature in df_2_.columns else np.array([])
                            t_stat = 0
                            p_val = 0
                            align = 0
                            differ = 0
                            # Only perform t-test if we have enough data
                            if len(data1) >= 2 and len(data2) >= 2:
                                try:
                                    # Levene test for variance homogeneity
                                    _, p_levene = stats.levene(data1, data2)
                                    # t-test with appropriate variance assumption
                                    if p_levene <= 0.05:
                                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                                        align = 0  # Unequal variance
                                    else:
                                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True)
                                        align = 1  # Equal variance
                                    differ = 1 if p_val <= 0.05 else 0
                                except:
                                    pass

                            feature_results.append({
                                'feature': feature,
                                'cls1': cls_ + str(i),
                                'cls2': cls__ + str(cls2_num),
                                'square_align': align,
                                't_stat': t_stat,
                                'p_value': p_val,
                                't_differ': differ
                            })

                        # 2. Feature mean vector comparison
                        # Initialize default values
                        stat_val = 0
                        p_value = 0
                        align_mean = 0
                        differ_mean = 0
                        test_type = "insufficient_data"
                        try:
                            n1 = df_1_.shape[0]
                            n2 = df_2_.shape[0]
                            p = len(f_list)
                            if n1 >= 2 and n2 >= 2 and n1 + n2 > p + 1:
                                # Hotelling's T² test for mean vectors
                                S1 = np.cov(df_1_.T, ddof=1)
                                S2 = np.cov(df_2_.T, ddof=1)
                                S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
                                mean_diff = df_1_mean_vector - df_2_mean_vector
                                inv_S_pooled = np.linalg.pinv(S_pooled)
                                T_squared = (n1 * n2 / (n1 + n2)) * np.dot(mean_diff.T, np.dot(inv_S_pooled, mean_diff))
                                F_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T_squared
                                df1, df2 = p, n1 + n2 - p - 1
                                p_value = 1 - stats.f.cdf(F_stat, df1, df2)
                                differ_mean = 1 if p_value <= 0.05 else 0
                                align_mean = 1  # Equal covariance assumed for Hotelling's test
                                test_type = "Hotelling_T2"
                                stat_val = F_stat
                            elif n1 >= 1 and n2 >= 1:
                                euclidean_dist = np.linalg.norm(df_1_mean_vector - df_2_mean_vector)
                                combined_data = np.vstack([df_1_mean_vector.reshape(1, -1),
                                                           df_2_mean_vector.reshape(1, -1)])
                                n_perm = 100
                                perm_dists = []
                                for _ in range(n_perm):
                                    perm_indices = np.random.permutation(2)
                                    perm_dist = np.linalg.norm(combined_data[perm_indices[0]] -
                                                               combined_data[perm_indices[1]])
                                    perm_dists.append(perm_dist)
                                p_value = np.sum(np.array(perm_dists) >= euclidean_dist) / n_perm
                                differ_mean = 1 if p_value <= 0.05 else 0
                                align_mean = 1
                                test_type = "permutation_test"
                                stat_val = euclidean_dist
                        except Exception:
                            pass

                        mean_results.append({
                            'cls1': cls_ + str(i),
                            'cls2': cls__ + str(cls2_num),
                            'n1': n1,
                            'n2': n2,
                            'n_features': len(f_list),
                            'test_type': test_type,
                            'statistic': stat_val,
                            'p_value': p_value,
                            'square_align': align_mean,
                            't_differ': differ_mean
                        })

                else:
                    # Between different subtypes
                    num_ = cls_num[cls__]
                    for jj in range(num_):
                        cls2_num = jj
                        df_2_ = df[df[cls__] == cls2_num][f_list]
                        if df_2_.shape[0] > 0:
                            df_2_mean_vector = df_2_.mean(axis=0).values
                            n2 = df_2_.shape[0]
                        else:
                            df_2_mean_vector = np.zeros(len(f_list))
                            n2 = 0

                        # 1. Per-feature analysis
                        for feature in f_list:
                            # Try to get data for t-test
                            data1 = df_1_[feature].dropna().values if feature in df_1_.columns else np.array([])
                            data2 = df_2_[feature].dropna().values if feature in df_2_.columns else np.array([])
                            # Initialize default values (zeros/NaN)
                            t_stat = 0
                            p_val = 0
                            align = 0
                            differ = 0
                            # Only perform t-test if we have enough data
                            if len(data1) >= 2 and len(data2) >= 2:
                                try:
                                    # Levene test for variance homogeneity
                                    _, p_levene = stats.levene(data1, data2)
                                    # t-test with appropriate variance assumption
                                    if p_levene <= 0.05:
                                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                                        align = 0  # Unequal variance
                                    else:
                                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True)
                                        align = 1  # Equal variance
                                    differ = 1 if p_val <= 0.05 else 0
                                except:
                                    pass

                            feature_results.append({
                                'feature': feature,
                                'cls1': cls_ + str(i),
                                'cls2': cls__ + str(cls2_num),
                                'square_align': align,
                                't_stat': t_stat,
                                'p_value': p_val,
                                't_differ': differ
                            })

                        # 2. Feature mean vector comparison - Always create row
                        # Initialize default values
                        stat_val = 0
                        p_value = 0
                        align_mean = 0
                        differ_mean = 0
                        test_type = "insufficient_data"
                        try:
                            n1 = df_1_.shape[0]
                            n2 = df_2_.shape[0]
                            p = len(f_list)
                            if n1 >= 2 and n2 >= 2 and n1 + n2 > p + 1:
                                # Hotelling's T² test for mean vectors
                                S1 = np.cov(df_1_.T, ddof=1)
                                S2 = np.cov(df_2_.T, ddof=1)
                                S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
                                mean_diff = df_1_mean_vector - df_2_mean_vector
                                inv_S_pooled = np.linalg.pinv(S_pooled)
                                T_squared = (n1 * n2 / (n1 + n2)) * np.dot(mean_diff.T, np.dot(inv_S_pooled, mean_diff))
                                F_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T_squared
                                df1, df2 = p, n1 + n2 - p - 1
                                p_value = 1 - stats.f.cdf(F_stat, df1, df2)
                                differ_mean = 1 if p_value <= 0.05 else 0
                                align_mean = 1  # Equal covariance assumed for Hotelling's test
                                test_type = "Hotelling_T2"
                                stat_val = F_stat

                            elif n1 >= 1 and n2 >= 1:
                                # Use Euclidean distance with permutation test
                                euclidean_dist = np.linalg.norm(df_1_mean_vector - df_2_mean_vector)
                                # Simple permutation test
                                combined_data = np.vstack([df_1_mean_vector.reshape(1, -1),
                                                           df_2_mean_vector.reshape(1, -1)])
                                n_perm = 100
                                perm_dists = []
                                for _ in range(n_perm):
                                    perm_indices = np.random.permutation(2)
                                    perm_dist = np.linalg.norm(combined_data[perm_indices[0]] -
                                                               combined_data[perm_indices[1]])
                                    perm_dists.append(perm_dist)
                                p_value = np.sum(np.array(perm_dists) >= euclidean_dist) / n_perm
                                differ_mean = 1 if p_value <= 0.05 else 0
                                align_mean = 1
                                test_type = "permutation_test"
                                stat_val = euclidean_dist
                        except Exception:
                            pass

                        mean_results.append({
                            'cls1': cls_ + str(i),
                            'cls2': cls__ + str(cls2_num),
                            'n1': n1,
                            'n2': n2,
                            'n_features': len(f_list),
                            'test_type': test_type,
                            'statistic': stat_val,
                            'p_value': p_value,
                            'square_align': align_mean,
                            't_differ': differ_mean
                        })

        if feature_results:
            feature_df = pd.DataFrame(feature_results)
            feature_df.to_excel(writer, sheet_name=cls_, index=False)
        if mean_results:
            mean_df = pd.DataFrame(mean_results)
            mean_df.to_excel(writer, sheet_name=cls_ + "_mean", index=False)