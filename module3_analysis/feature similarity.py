'''

Feature similarity

Compute the similarity of feature spaces between subtypes (or morphotypes).

Includes several metrics:
1. Manifold overlap
2. Statistical distribution
3. Separability score (classification)
4. Feature importance
5. JS divergence
6. Overall similarity
7. KDE analysis (plot + overlap_ratio)

An example of the operation can be found in the main function.

'''



import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, jensenshannon
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import KDTree
from contextlib import redirect_stdout
import os

plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

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
feature_names = ['shape_VoxelVolume',
                 'shape_MeshVolume',
                 'shape_SurfaceArea',
                 'shape_SurfaceVolumeRatio',
                 'shape_Compactness1',
                 'shape_Compactness2',
                 'shape_Sphericity',
                 'shape_SphericalDisproportion',
                 'shape_Maximum3DDiameter',
                 'shape_Maximum2DDiameterSlice',
                 'shape_Maximum2DDiameterColumn',
                 'shape_Maximum2DDiameterRow',
                 'shape_MajorAxisLength',
                 'shape_MinorAxisLength',
                 'shape_LeastAxisLength',
                 'shape_Elongation',
                 'shape_Flatness',
                 'glcm_Autocorrelation',
                 'glcm_JointAverage',
                 'glcm_ClusterProminence',
                 'glcm_ClusterShade',
                 'glcm_ClusterTendency',
                 'glcm_Contrast',
                 'glcm_Correlation',
                 'glcm_JointEntropy',
                 'glcm_Imc1',
                 'glcm_MaximumProbability',
                 'glcm_SumEntropy',
                 'glcm_SumSquares',
                 'glcm_DifferenceAverage',
                 'glcm_DifferenceEntropy',
                 'glcm_DifferenceVariance',
                 'glcm_JointEnergy',
                 'glcm_Imc2',
                 'glcm_Idm',
                 'glcm_Idmn',
                 'glcm_Id',
                 'glcm_Idn',
                 'glcm_InverseVariance',
                 'glszm_GrayLevelVariance',
                 'glszm_ZoneVariance',
                 'glszm_SizeZoneNonUniformity',
                 'glszm_GrayLevelNonUniformity',
                 'glszm_SmallAreaHighGrayLevelEmphasis',
                 'glszm_LargeAreaLowGrayLevelEmphasis',
                 'glszm_LargeAreaHighGrayLevelEmphasis',
                 'glszm_ZonePercentage',
                 'glszm_ZoneEntropy',
                 'glszm_SmallAreaLowGrayLevelEmphasis',
                 'glszm_SmallAreaEmphasis',
                 'glszm_LargeAreaEmphasis',
                 'glszm_SizeZoneNonUniformityNormalized',
                 'glszm_GrayLevelNonUniformityNormalized',
                 'glszm_HighGrayLevelZoneEmphasis',
                 'glszm_LowGrayLevelZoneEmphasis',
                 'ngtdm_Coarseness',
                 'ngtdm_Contrast',
                 'ngtdm_Busyness',
                 'ngtdm_Complexity',
                 'ngtdm_Strength',
                 'gldm_SmallDependenceEmphasis',
                 'gldm_GrayLevelNonUniformity',
                 'gldm_LargeDependenceEmphasis',
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
                 'gldm_LargeDependenceHighGrayLevelEmphasis']


def create_master_figure_complete(group_a, group_b, feature_names=None,
                                  save_path=None, figsize=(18, 10)):

    results = {}

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[1.2, 1.1, 1.1], width_ratios=[1.1, 1.1, 1.1])

    # Manifold overlap diagram
    ax_a = fig.add_subplot(gs[0:2, 0])
    overlap_score = _create_manifold_subplot(fig, group_a, group_b, ax_a)
    results['manifold_overlap'] = overlap_score

    # Statistical chart
    ax_b = fig.add_subplot(gs[:, 1])
    stats_results = _create_statistical_subplot(fig, group_a, group_b, ax_b, feature_names)
    results.update(stats_results)

    # Decision diagram
    ax_c = fig.add_subplot(gs[2, 0])
    separability_score = _create_decision_subplot(fig, group_a, group_b, ax_c)
    results['separability_score'] = separability_score

    # Feature Importance Plot
    ax_d = fig.add_subplot(gs[2, 2])
    _create_feature_importance_subplot(fig, group_a, group_b, ax_d, feature_names)

    # Distribution Overlay Heatmap
    ax_e = fig.add_subplot(gs[0, 2])
    js_divergence = _create_distribution_heatmap(fig, group_a, group_b, ax_e)
    results['js_divergence'] = js_divergence

    # Sample Distance Network Diagram
    ax_f = fig.add_subplot(gs[1, 2])
    separation_index = _create_distance_network_subplot(fig, group_a, group_b, ax_f)
    results['separation_index'] = separation_index

    # Calculate the overall similarity score
    overall_similarity = (1 - results.get('js_divergence', 0) +
                          results.get('manifold_overlap', 0)) / 2
    results['overall_similarity'] = overall_similarity

    plt.suptitle(
        'Comprehensive Analysis of Distribution Similarity\n'
        f'Between Mouse (n={len(group_a)}) and AD (n={len(group_b)})',
        fontsize=16, y=0.98, weight='bold'
    )

    summary_text = (
        f"Overall Similarity Score: {overall_similarity:.2%}\n"
        f"Manifold Overlap: {results.get('manifold_overlap', 0):.1%}\n"
        f"JS Divergence: {results.get('js_divergence', 0):.3f}\n"
        f"Separability: {results.get('separability_score', 0):.2f}\n"
        f"Separation Index: {results.get('separation_index', 0):.2f}"
    )

    ax_text = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, summary_text, ha='center', va='center',
                 fontsize=11, linespacing=1.5,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3, pad=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.tight_layout(rect=[0, 0.1, 1, 0.97])

    return fig, results



def _create_manifold_subplot(fig, group_a, group_b, ax):
    #  1. Merge data and reduce dimensionality
    combined = np.vstack([group_a, group_b])

    # UMAP
    n_neighbors = min(15, len(combined) // 10)
    if n_neighbors <= 1:
        n_neighbors = 2
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(combined)

    a_points = embedding[:len(group_a)]
    b_points = embedding[len(group_a):]

    # 2. Plot a scatter diagram
    ax.scatter(a_points[:, 0], a_points[:, 1],
               c='#1E88E5', alpha=0.6, s=40,
               edgecolors='white', linewidth=0.5,
               label=f'Group A (n={len(group_a)})')

    ax.scatter(b_points[:, 0], b_points[:, 1],
               c='#D81B60', alpha=0.6, s=40,
               edgecolors='white', linewidth=0.5,
               label=f'Group B (n={len(group_b)})')

    # 3. Calculating Overlap: KDE Analysis
    tree_a = KDTree(a_points)
    tree_b = KDTree(b_points)

    dist_a_to_b, _ = tree_b.query(a_points, k=1)
    dist_b_to_a, _ = tree_a.query(b_points, k=1)

    threshold = np.median(np.concatenate([dist_a_to_b, dist_b_to_a])) * 0.5
    overlap_a = np.mean(dist_a_to_b < threshold)
    overlap_b = np.mean(dist_b_to_a < threshold)
    overlap_score = (overlap_a + overlap_b) / 2

    ax.set_xlabel('UMAP Dimension 1', fontsize=10)
    ax.set_ylabel('UMAP Dimension 2', fontsize=10)
    ax.set_title(f'Manifold Projection\nOverlap: {overlap_score:.1%}',
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')

    return overlap_score


def _create_statistical_subplot(fig, group_a, group_b, ax_main, feature_names=None):
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax_main.get_subplotspec(),
                                       hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs_inner[0, 0])
    ax2 = fig.add_subplot(gs_inner[0, 1])
    ax3 = fig.add_subplot(gs_inner[1, 0])
    ax4 = fig.add_subplot(gs_inner[1, 1])

    results = {}

    # Panel 1: Effect Size Forest Plot
    effect_sizes = []

    n_features = group_a.shape[1]
    if feature_names is None:
        feature_names = [f'F{i + 1}' for i in range(n_features)]

    display_features = min(10, n_features)
    # display_features = n_features

    group_a = group_a.values
    group_b = group_b.values

    for i in range(display_features):
        mean_a, mean_b = np.mean(group_a[:, i]), np.mean(group_b[:, i])
        std_a, std_b = np.std(group_a[:, i]), np.std(group_b[:, i])
        n_a, n_b = len(group_a), len(group_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
        d = (mean_a - mean_b) / pooled_std
        effect_sizes.append(d)

        color = '#D81B60' if abs(d) > 0.5 else '#1E88E5'
        ax1.plot(d, i, 'o', color=color, markersize=6)
        ci_width = 1.96 / np.sqrt(min(n_a, n_b))
        ax1.plot([d - ci_width, d + ci_width], [i, i], color=color, linewidth=1.5, alpha=0.7)

    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.2)
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.2)
    ax1.set_yticks(range(display_features))
    ax1.set_yticklabels(feature_names[:display_features], fontsize=8)
    ax1.set_xlabel("Cohen's d", fontsize=9)
    ax1.set_title('Effect Sizes', fontsize=11)
    ax1.set_xlim(-2, 2)
    ax1.grid(True, alpha=0.2, axis='x')

    results['mean_effect_size'] = np.mean(np.abs(effect_sizes))

    # Panel 2: Distance Distribution
    # sample
    n_samples_a = min(5000, len(group_a))
    n_samples_b = min(5000, len(group_b))
    # n_samples_a = len(group_a)
    # n_samples_b = len(group_b)

    within_a = pdist(group_a[:n_samples_a], metric='euclidean')
    within_b = pdist(group_b[:n_samples_b], metric='euclidean')

    # n_between_samples = min(100, n_samples_a * n_samples_b)
    n_between_samples = n_samples_a * n_samples_b
    indices_a = np.random.choice(n_samples_a, n_between_samples, replace=True)
    indices_b = np.random.choice(n_samples_b, n_between_samples, replace=True)
    between = [np.linalg.norm(group_a[i] - group_b[j])
               for i, j in zip(indices_a[:n_between_samples], indices_b[:n_between_samples])]

    ax2.hist(within_a, bins=20, alpha=0.5, density=True,
             color='#1E88E5', label='Within A')
    ax2.hist(within_b, bins=20, alpha=0.5, density=True,
             color='#D81B60', label='Within B')
    ax2.hist(between, bins=20, alpha=0.5, density=True,
             color='#4CAF50', label='Between A-B')
    ax2.set_xlabel('Distance', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.set_title('Distance Distributions', fontsize=11)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.2)

    # Calculate the separation index
    separation_index = np.mean(between) / ((np.mean(within_a) + np.mean(within_b)) / 2)
    results['separation_index'] = separation_index
    ax2.text(0.05, 0.95, f'Separation Index: {separation_index:.2f}',
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')

    # Panel 3: Differences in Correlation Matrix
    # Calculate the correlation matrix (using all features, but display the top 10).
    display_corr = min(10, n_features)
    # display_corr = n_features
    corr_a = np.corrcoef(group_a[:, :display_corr].T)
    corr_b = np.corrcoef(group_b[:, :display_corr].T)
    corr_diff = np.abs(corr_a - corr_b)

    im = ax3.imshow(corr_diff, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    ax3.set_title(f'Correlation Difference\n(Top {display_corr} features)', fontsize=11)
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4: Cumulative Distribution Function
    # Select the top five features.
    n_cdf_features = min(5, n_features)
    # n_cdf_features = n_features
    for i in range(n_cdf_features):
        sorted_a = np.sort(group_a[:, i])
        sorted_b = np.sort(group_b[:, i])
        cdf_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
        cdf_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)

        ax4.plot(sorted_a, cdf_a, alpha=0.7, linewidth=1.5)
        ax4.plot(sorted_b, cdf_b, alpha=0.7, linewidth=1.5, linestyle='--')

    ax4.set_xlabel('Feature Value', fontsize=9)
    ax4.set_ylabel('Cumulative Probability', fontsize=9)
    ax4.set_title('CDF Comparison', fontsize=11)
    ax4.legend([f'Feat {i + 1}' for i in range(n_cdf_features)], fontsize=8)
    ax4.grid(True, alpha=0.2)

    ax_main.set_axis_off()

    return results


def create_safe_meshgrid(X, y=None, max_points=100):
    if X.shape[1] != 2:
        raise ValueError("X must have 2 dimensions")

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)

    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    x_range = x_max - x_min
    y_range = y_max - y_min

    h_x = max(x_range / max_points, 0.01)
    h_y = max(y_range / max_points, 0.01)
    h = max(h_x, h_y)

    x_points = int(x_range / h) + 1
    y_points = int(y_range / h) + 1

    x_points = min(x_points, max_points)
    y_points = min(y_points, max_points)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, x_points),
                         np.linspace(y_min, y_max, y_points))

    return xx, yy


def _create_decision_subplot(fig, group_a, group_b, ax):
    X = np.vstack([group_a, group_b])
    y = np.array([0] * len(group_a) + [1] * len(group_b))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # print(f"PCA data range:")
    # print(f"  X: [{X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}]")
    # print(f"  Y: [{X_pca[:, 1].min():.2f}, {X_pca[:, 1].max():.2f}]")

    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    scores = cross_val_score(clf, X_pca, y, cv=5)
    accuracy = scores.mean()

    xx, yy = create_safe_meshgrid(X_pca, max_points=150)

    clf.fit(X_pca, y)

    if xx.size > 10000:
        Z = np.zeros(xx.ravel().shape)
        batch_size = 10000

        for i in range(0, len(Z), batch_size):
            end_idx = min(i + batch_size, len(Z))
            points = np.c_[xx.ravel()[i:end_idx], yy.ravel()[i:end_idx]]
            Z[i:end_idx] = clf.predict(points)

        Z = Z.reshape(xx.shape)
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1E88E5', '#D81B60'])

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    max_display_points = 500
    if len(X_pca) > max_display_points:
        indices = np.random.choice(len(X_pca), max_display_points, replace=False)
        X_display = X_pca[indices]
        y_display = y[indices]
    else:
        X_display = X_pca
        y_display = y

    ax.scatter(X_display[y_display == 0, 0], X_display[y_display == 0, 1],
               color='#1E88E5', alpha=0.7, s=30,
               edgecolors='white', label='Group A')
    ax.scatter(X_display[y_display == 1, 0], X_display[y_display == 1, 1],
               color='#D81B60', alpha=0.7, s=30,
               edgecolors='white', label='Group B')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
    ax.set_title(f'Decision Boundaries\nCV Accuracy: {accuracy:.2f}',
                 fontsize=11, pad=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    return accuracy


def _create_feature_importance_subplot(fig, group_a, group_b, ax, feature_names=None):
    from sklearn.ensemble import RandomForestClassifier

    X = np.vstack([group_a, group_b])
    y = np.array([0] * len(group_a) + [1] * len(group_b))

    # Training a Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = rf.feature_importances_
    n_features = group_a.shape[1]

    n_display = min(10, n_features)     # Select the top 10 most important features.
    # n_display = n_features
    indices = np.argsort(importance)[-n_display:]

    if feature_names is None:
        feature_names = [f'Feature {i + 1}' for i in range(n_features)]

    ax.barh(range(n_display), importance[indices],
            color='#4CAF50', alpha=0.7)
    ax.set_yticks(range(n_display))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel('Importance', fontsize=9)
    ax.set_title(f'Top {n_display} Important Features\nfor Classification',
                 fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)


def _create_distribution_heatmap(fig, group_a, group_b, ax):
    # PCA
    pca = PCA(n_components=1)
    a_pca = pca.fit_transform(group_a)
    b_pca = pca.fit_transform(group_b)

    # KDE
    grid = np.linspace(-3, 3, 100)
    kde_a = gaussian_kde(a_pca[:, 0])
    kde_b = gaussian_kde(b_pca[:, 0])
    pdf_a = kde_a(grid)
    pdf_b = kde_b(grid)
    pdf_a = pdf_a / pdf_a.sum()
    pdf_b = pdf_b / pdf_b.sum()

    # JSD
    js_div = jensenshannon(pdf_a, pdf_b)

    ax2 = ax.twinx()

    ax.plot(grid, pdf_a, color='#1E88E5', linewidth=2, alpha=0.8, label='Group A')
    ax2.plot(grid, pdf_b, color='#D81B60', linewidth=2, alpha=0.8, label='Group B')

    ax.fill_between(grid, pdf_a, alpha=0.3, color='#1E88E5')
    ax2.fill_between(grid, pdf_b, alpha=0.3, color='#D81B60')

    ax.set_xlabel('Principal Component 1', fontsize=9)
    ax.set_ylabel('Density (Group A)', color='#1E88E5', fontsize=9)
    ax2.set_ylabel('Density (Group B)', color='#D81B60', fontsize=9)

    ax.set_title(f'Distribution Comparison\nJS Divergence: {js_div:.3f}',
                 fontsize=11, pad=10)

    ax.tick_params(axis='y', labelcolor='#1E88E5')
    ax2.tick_params(axis='y', labelcolor='#D81B60')

    return js_div


def _create_distance_network_subplot(fig, group_a, group_b, ax):
    n_samples = min(5000, len(group_a), len(group_b))
    # n_samples = min(len(group_a), len(group_b))
    indices_a = np.random.choice(len(group_a), n_samples, replace=False)
    indices_b = np.random.choice(len(group_b), n_samples, replace=False)

    sample_a = group_a.iloc[indices_a]
    sample_b = group_b.iloc[indices_b]

    from scipy.spatial.distance import cdist
    distances = cdist(sample_a, sample_b, metric='euclidean')

    im = ax.imshow(distances, cmap='viridis', aspect='auto')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(-0.5, n_samples / 2, 'Group A', ha='right', va='center',
            rotation=90, fontsize=9, transform=ax.transData)
    ax.text(n_samples / 2, -0.5, 'Group B', ha='center', va='top',
            fontsize=9, transform=ax.transData)

    ax.set_title('Pairwise Distance Matrix', fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    within_a = pdist(sample_a, metric='euclidean')
    within_b = pdist(sample_b, metric='euclidean')
    between = distances.ravel()

    separation_index = np.mean(between) / ((np.mean(within_a) + np.mean(within_b)) / 2)

    ax.text(0.05, 0.95, f'Avg Between: {np.mean(between):.2f}\n'
                        f'Avg Within: {(np.mean(within_a) + np.mean(within_b)) / 2:.2f}\n'
                        f'Separation: {separation_index:.2f}',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')

    return separation_index


def add_region(df):
    df_area = pd.read_excel('./test/672areasvolume.xlsx')  # Replace region with the corresponding number from the table
    region_num = []
    region_big = []
    for i in range(df.shape[0]):
        region = df.iloc[i]["region"][1:-1]
        r_num = int(df_area[df_area['Region_small'] == region]["num"].values[0])
        r_big = int(df_area[df_area['Region_small'] == region]["num_big"].values[0])
        region_num.append(r_num)
        region_big.append(r_big)
    df["region"] = region_num
    df.insert(loc=df.shape[1], column='region_big', value=region_big)
    return df



COLOR_SCHEME = {'mouse': '#C0C0C0', 'ad_primary': '#6495ED'}
def create_kde_plot_for_mouse_cluster(mouse_data, ad_data, mouse_cls, ad_cls, output_dir):
    all_data = []
    mouse_values = []
    ad_values = []

    for feature in feature_names_2D:
        values = mouse_data[feature].dropna().values
        if len(values) >= 5:
            mouse_values.extend(values)
    for feature in feature_names_2D:
        values = ad_data[feature].dropna().values
        if len(values) >= 5:
            ad_values.extend(values)

    all_data.append({
        'label': f'M{mouse_cls + 1}',
        'values': np.array(mouse_values),
        'color': COLOR_SCHEME['mouse']
    })
    all_data.append({
        'label': f'A{ad_cls + 1}',
        'values': np.array(ad_values),
        'color': COLOR_SCHEME['ad_primary']
    })

    fig, ax = plt.subplots(figsize=(6, 8))

    all_values = []
    for data in all_data:
        all_values.extend(data['values'])
    all_values = np.array(all_values)
    x_min = all_values.min()
    x_max = all_values.max()
    x_range = np.linspace(x_min, x_max, 1000)
    for data in all_data:
        try:
            kde = gaussian_kde(data['values'])
            kde_curve = kde(x_range)
            ax.plot(x_range, kde_curve,
                    label=data['label'],
                    linewidth=2,
                    color=data['color'],
                    alpha=0.8)
            ax.fill_between(x_range, kde_curve,
                            alpha=0.3,
                            color=data['color'])
        except Exception as e:
            print(f"{data['label']} error: {e}")
            continue

    kde_mouse = gaussian_kde(all_data[0]['values'])
    kde_ad = gaussian_kde(all_data[1]['values'])

    kde_mouse_curve = kde_mouse(x_range)
    kde_ad_curve = kde_ad(x_range)

    overlap_area = np.trapz(np.minimum(kde_mouse_curve, kde_ad_curve), x_range)
    total_area = np.trapz(kde_mouse_curve + kde_ad_curve, x_range) / 2
    overlap_ratio = overlap_area / total_area if total_area > 0 else 0

    ax.grid(True, alpha=0.5, axis='y', linestyle='--', linewidth=1.0)
    ax.axis([-0.05, 1.05, -0.5787150395204081, 12.15301582992857])
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_major_formatter('{x:.1f}')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=11,
                   length=8,
                   width=1.5,
                   pad=5)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'kde_M{mouse_cls + 1}_M{ad_cls + 1}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # output_trans_path = os.path.join(output_dir, f'kde_M{mouse_cls + 1}_M{ad_cls + 1}-trans.png')
    # plt.savefig(output_trans_path, dpi=300, transparent=True, bbox_inches='tight')

    plt.yscale('log')
    plt.ylim(-0.5787150395204081, 12.15301582992857)
    output_path = os.path.join(output_dir, f'kde_M{mouse_cls + 1}_M{ad_cls + 1}-log.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # output_trans_path = os.path.join(output_dir, f'kde_M{mouse_cls + 1}_M{ad_cls + 1}-log-trans.png')
    # plt.savefig(output_trans_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

    if len(all_data) == 2:
        return {
            'mouse_cluster': f'M{mouse_cls + 1}',
            'ad_clusters': f'A{ad_cls + 1}',
            'overlap_ratio': overlap_ratio,
            'mouse_samples': len(all_data[0]['values']),
            'ad_samples': len(all_data[1]['values']),
        }
    return None



out_dir = "./test/output/"
os.makedirs(out_dir, exist_ok=True)

cls_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
cls_2_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

df = pd.read_excel("./test/test_2D.xlsx")

KDE_results = []

for cls in cls_list:
    for cls2 in cls_2_list:
        AD = df[df["multi"] == cls2][feature_names_2D]   # Select the appropriate feature set based on practical requirements: 2D or 3D.
        # if (AD.shape[0] > 60000) & (AD.shape[0] <= 150000):
        #     AD = AD.sample(frac=0.7, random_state=42)
        # if (AD.shape[0] > 150000) & (AD.shape[0] <= 1000000):
        #     AD = AD.sample(frac=0.5, random_state=42)
        # if AD.shape[0] > 1000000:
        #     AD = AD.sample(frac=0.3, random_state=42)
        # if AD.shape[0] <= 1:
        #     continue
        if AD.shape[0] <= 5:
            continue

        mouse = df[df["multi"] == cls][feature_names_2D]
        # if (mouse.shape[0] > 60000) & (mouse.shape[0] <= 150000):
        #     mouse = mouse.sample(frac=0.7, random_state=42)
        # if (mouse.shape[0] > 150000) & (mouse.shape[0] <= 1000000):
        #     mouse = mouse.sample(frac=0.5, random_state=42)
        # if mouse.shape[0] > 1000000:
        #     mouse = mouse.sample(frac=0.3, random_state=42)
        # if mouse.shape[0] <= 1:
        #     continue

        # Create the master figure
        if mouse.shape[0] <= 5:
            continue

        fig, results = create_master_figure_complete(
            mouse, AD,
            feature_names=feature_names_2D,
            save_path=out_dir + "M" + str(cls+1) + "-M" + str(cls2+1) + '.png',
            figsize=(30, 20)
        )

        with open(out_dir + "M" + str(cls+1) + "-M" + str(cls2+1) + '.txt', 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("\n" + "=" * 60)
                print("ANALYSIS RESULTS SUMMARY")
                print("=" * 60)
                for key, value in results.items():
                    print(f"{key:20s}: {value:.4f}")

                print("\n" + "=" * 60)
                print("Interpretation of Results:")
                print("=" * 60)
                print(f"1. manifold_overlap: {results.get('manifold_overlap', 0):.1%}")
                print("   - A higher value indicates greater overlap between the two groups in the feature space.")
                print(f"\n2. js_divergence: {results.get('js_divergence', 0):.3f}")
                print("   - The value lies between 0 and 1, where 0 indicates identical distributions and 1 indicates completely different distributions.")
                print(f"\n3. separation_index: {results.get('separation_index', 0):.2f}")
                print("   - >1.5: forced separation, 1.2-1.5: moderate separation, <1.2: weak separation")
                print(f"\n4. separability_score: {results.get('separability_score', 0):.2f}")
                print("   - 0.5: random classification, 1.0: good")
                print(f"\n5. mean_effect_size: {results.get('mean_effect_size', 0):.2f}")
                print("   - This value lies between 0 and 1, where 0 indicates similarity and 1 indicates difference.")
                print(f"\n6. overall_similarity: {results.get('overall_similarity', 0):.2f}")
                print("   - The value ranges from 0 to 100%, with 100% indicating complete identity.")

        result = create_kde_plot_for_mouse_cluster(mouse, AD, cls, cls2, out_dir)
        if result:
            KDE_results.append(result)


if KDE_results:
    summary_data = []
    for result in KDE_results:
        row = {
            'Subtype1': result['mouse_cluster'],
            'Subtype2': result['ad_clusters'],
            'overlap_ratio': f"{result['overlap_ratio']:.2%}"
        }
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(out_dir, 'kde_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
