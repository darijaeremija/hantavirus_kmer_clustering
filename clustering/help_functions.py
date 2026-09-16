import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
import os
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from IPython.display import display



def build_kmer_dataset(df, k, file_name, output_dir='../data/sequences_kmers'):
    os.makedirs(output_dir, exist_ok=True)
    
    bases = ['A', 'C', 'G', 'T']
    
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    
    n_samples = len(df)
    n_features = 4 ** k
    
    matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    
    sequences = df['Sequence'].values
    for row_idx, seq in enumerate(sequences):
        seq_len = len(seq)
        total_kmers = seq_len - k + 1
        
        if total_kmers <= 0:
            continue
            
        for i in range(total_kmers):
            kmer = seq[i:i+k]
            idx = kmer_to_idx.get(kmer)
            if idx is not None:
                matrix[row_idx, idx] += 1.0
        
        # L1 normalizacija
        row_sum = matrix[row_idx].sum()
        if row_sum > 0:
            matrix[row_idx] /= row_sum
    df_kmers = pd.DataFrame(matrix, columns=all_kmers)

    df_final = pd.concat([df.reset_index(drop=True), df_kmers], axis=1)
    
    path = os.path.join(output_dir, file_name)
    df_final.to_csv(path, index=False)
    
    return df_final

def plot_cumulative_variance(df, dataset_label='Dataset'):
    
    cols_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'Accession', 'Species','Segment', 'Nuc_Completeness', 'Sequence']

    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    n_components = len(cum_var)
    
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, n_components + 1), cum_var, marker='o', markersize=2, linestyle='-', label='Kumulativna varijansa')
    
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Prag 80%')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Prag 90%')
    
    plt.xlabel('Broj glavnih komponenti (PCA)', fontsize=11)
    plt.ylabel('Objašnjena varijansa (%)', fontsize=11)
    plt.title(f'Kumulativna objašnjena varijansa - {dataset_label}', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.show()
    
    n_80 = np.argmax(cum_var >= 80) + 1
    n_90 = np.argmax(cum_var >= 90) + 1
    print(f'[{dataset_label}] Broj komponenti za 80% varijanse: {n_80}')
    print(f'[{dataset_label}] Broj komponenti za 90% varijanse: {n_90}')

def apply_pca_transformation(df, n_components=0.8):

    cols_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'Accession', 'Species','Segment', 'Nuc_Completeness', 'Sequence']

    X = df.drop(columns=cols_to_drop, errors='ignore')
    species = df['Species'].values
    segments = df['Segment'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, species, segments

def optimize_and_print_kmeans(X_pca, cluster_range=range(2, 9)):
    
    best_score = -1
    best_params = {}
    best_labels = None
    best_metrics = {}
    
    inits = ['k-means++', 'random']
    
    for k in cluster_range:
        for init_method in inits:
            kmeans = KMeans(
                n_clusters=k, 
                init=init_method, 
                 n_init=10, 
                random_state=42
            )
            labels = kmeans.fit_predict(X_pca)
                
            sil_score = silhouette_score(X_pca, labels)
            db_score = davies_bouldin_score(X_pca, labels)
                
            if sil_score > best_score:
                best_score = sil_score
                best_params = {
                    'n_clusters': k,
                    'init': init_method
                }
                best_labels = labels
                best_metrics = {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score
                }
                    
    print('=== NAJBOLJI K-MEANS REZULTATI ===')
    print(f"Optimalan broj klastera: {best_params['n_clusters']}")
    print(f"Način inicijalizacije: {best_params['init']}")
    print('-' * 40)
    print(f"Silueta koeficijent: {best_metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin indeks: {best_metrics['davies_bouldin']:.4f}")
    
    return best_params, best_metrics, best_labels

def optimize_and_print_agglomerative(X_pca, cluster_range=range(2, 9), linkages=['ward', 'complete', 'average', 'single'], metrics_list=['euclidean', 'manhattan', 'cosine']):
   
    best_score = -1
    best_params = {}
    best_labels = None
    best_metrics = {}
        
    for k in cluster_range:
        for metric in metrics_list:
            for linkage in linkages:
                # Ward zahteva isključivo euklidsku distancu
                if linkage == 'ward' and metric != 'euclidean':
                    continue
                
                model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage)
                labels = model.fit_predict(X_pca)
                    
                sil_score = silhouette_score(X_pca, labels)
                db_score = davies_bouldin_score(X_pca, labels)
                    
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = {
                        'n_clusters': k,
                        'metric': metric,
                        'linkage': linkage
                    }
                    best_labels = labels
                    best_metrics = {
                        'silhouette': sil_score,
                        'davies_bouldin': db_score
                    }
                    
    print('=== NAJBOLJI AGGLOMERATIVE REZULTATI ===')
    print(f"Optimalan broj klastera: {best_params.get('n_clusters', 'N/A')}")
    print(f"Izabrana metrika: {best_params.get('metric', 'N/A')}")
    print(f"Tip povezivanja: {best_params.get('linkage', 'N/A')}")
    print('-' * 40)
    print(f"Silueta koeficijent: {best_metrics.get('silhouette', 0):.4f}")
    print(f"Davies-Bouldin indeks: {best_metrics.get('davies_bouldin', 0):.4f}")
    
    return best_params, best_metrics, best_labels

def optimize_and_print_gmm(X_pca, cluster_range=range(2, 9), reg_covar=1e-6):
    best_score = -1
    best_params = {}
    best_labels = None
    best_metrics = {}
    
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    
    for k in cluster_range:
        for cov_type in covariance_types:
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type,reg_covar=reg_covar, random_state=42)
            labels = gmm.fit_predict(X_pca)
                
            sil_score = silhouette_score(X_pca, labels)
            db_score = davies_bouldin_score(X_pca, labels)
                
            if sil_score > best_score:
                best_score = sil_score
                best_params = {
                    'n_components': k,
                    'covariance_type': cov_type
                }
                best_labels = labels
                best_metrics = {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score
                }
                
    print('=== NAJBOLJI GMM REZULTATI ===')
    print(f"Optimalan broj klastera: {best_params.get('n_components', 'N/A')}")
    print(f"Tip kovarijanse: {best_params.get('covariance_type', 'N/A')}")
    print('-' * 40)
    print(f"Silueta koeficijent: {best_metrics.get('silhouette', 0):.4f}")
    print(f"Davies-Bouldin indeks: {best_metrics.get('davies_bouldin', 0):.4f}")
    
    return best_params, best_metrics, best_labels

def plot_3d(X, data, title_name):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pretvaranje stringova u brojeve za bojenje
    color_codes, uniques = pd.factorize(data)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color_codes, cmap='tab10', s=35, alpha=0.8)
    
    ax.set_title(f'3D PCA - {title_name}', fontsize=14)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Automatska legenda
    handles, _ = scatter.legend_elements(prop='colors', alpha=0.8)
    ax.legend(handles, uniques, title=title_name, loc='best')
    
    plt.tight_layout()
    plt.show()


def plot_clustering_comparison_3d(X_pca, kmeans_labels, agg_labels, gmm_labels):
    
    fig = plt.figure(figsize=(18, 6))

    k_kmeans = len(np.unique(kmeans_labels))
    k_agg = len(np.unique(agg_labels))
    k_gmm = len(np.unique(gmm_labels))

    # 1. K-Means
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans_labels, cmap='viridis', s=20, alpha=0.8)
    ax1.set_title(f'K-Means (k={k_kmeans})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    
    handles1, labels1 = scatter1.legend_elements()
    ax1.legend(handles1, labels1, title="Klasteri", loc="upper right")

    # 2. Agglomerative
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=agg_labels, cmap='viridis', s=20, alpha=0.8)
    ax2.set_title(f'Agglomerative (k={k_agg})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    
    handles2, labels2 = scatter2.legend_elements()
    ax2.legend(handles2, labels2, title="Klasteri", loc="upper right")

    # 3. GMM
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=gmm_labels, cmap='viridis', s=20, alpha=0.8)
    ax3.set_title(f'GMM (k={k_gmm})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    
    handles3, labels3 = scatter3.legend_elements()
    ax3.legend(handles3, labels3, title="Klasteri", loc="upper right")

    plt.tight_layout()
    plt.show()
    

def run_hantavirus_clustering_pipeline(complete_data, n_components=0.8, plot = False, reg_covar=1e-6):
    print("1. Pokretanje PCA redukcije dimenzionalnosti...")
    X_pca, pca, species, segments = apply_pca_transformation(complete_data, n_components)
    
    print("2. Izvršavanje K-Means klasterovanja...")
    kmeans_params, kmeans_metrics, kmeans_labels = optimize_and_print_kmeans(X_pca, cluster_range=range(2, 9))
    
    print('*' * 40)

    print("3. Izvršavanje Agglomerative klasterovanja...")
    agg_params, agg_metrics, agg_labels = optimize_and_print_agglomerative(X_pca, cluster_range=range(2, 9))

    
    print('*' * 40)

    print("4. Izvršavanje GMM klasterovanja...")
    gmm_params, gmm_metrics, gmm_labels = optimize_and_print_gmm(X_pca, cluster_range=range(2, 9),reg_covar=reg_covar)

    if plot:
        print("5. Generisanje uporednog 3D prikaza...")
        plot_clustering_comparison_3d(X_pca, kmeans_labels, agg_labels, gmm_labels)
    
    return {
        'pca': pca,
        'X_pca': X_pca,
        'species': species,  
        'segments': segments,
        'labels': {
            'kmeans': kmeans_labels,
            'agglomerative': agg_labels,
            'gmm': gmm_labels
        },
        'params': {  
            'kmeans': kmeans_params,
            'agglomerative': agg_params,
            'gmm': gmm_params
        },
        'metrics': {  
            'kmeans': kmeans_metrics,
            'agglomerative': agg_metrics,
            'gmm': gmm_metrics
        }
    }


def analyze_clustering_results(cluster_labels, data, segments, species, model_name="Model", top_n=3):

    cols_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'Accession', 'Species', 'Segment', 'Nuc_Completeness', 'Sequence']
    X_features = data.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"\n{'='*25} ANALIZA ZA: {model_name.upper()} {'='*25}")
    
    df_analysis = X_features.copy()
    df_analysis['Cluster'] = cluster_labels
    df_analysis['Segment'] = segments.values if hasattr(segments, 'values') else segments
    df_analysis['Species'] = species.values if hasattr(species, 'values') else species
    
    print(f"\n--- 1. Unakrsna tabela: Klasteri i genetski segmenti (L, M, S) ---")
    seg_crosstab = pd.crosstab(df_analysis['Cluster'], df_analysis['Segment'])
    display(seg_crosstab)
    
    print(f"\n--- 2. Unakrsna tabela: Klasteri i vrste hantavirusa ---")
    species_crosstab = pd.crosstab(df_analysis['Cluster'], df_analysis['Species'])
    display(species_crosstab)
    
    print("\n--- Dominantne vrste po klasterima ---")
    for cluster_id in sorted(df_analysis['Cluster'].unique()):
        sub = df_analysis[df_analysis['Cluster'] == cluster_id]
        total_in_c = len(sub)
        dominant_species = sub['Species'].value_counts()
        
        # Prikaz najzastupljenijih vrsta
        top_sp_str = ", ".join([f"{sp}: {cnt} ({cnt/total_in_c*100:.1f}%)" for sp, cnt in dominant_species.head(3).items()])
        dominant_seg = sub['Segment'].mode()[0]
        print(f"  Klaster {cluster_id} (N={total_in_c}, pretežno {dominant_seg} segment) -> {top_sp_str}")
        
    print(f"\n--- 3. Top {top_n} k-mera po Z-skoru po klasterima ---")
    
    feature_cols = [col for col in X_features.columns]
    
    # Računanje globalnog proseka i standardne devijacije
    global_mean = X_features[feature_cols].mean()
    global_std = X_features[feature_cols].std().replace(0, 1e-8)
    
    # Prosečne frekvencije po klasterima
    cluster_profiles = df_analysis.groupby('Cluster')[feature_cols].mean()
    
    # Z-skor profili
    cluster_zscores = (cluster_profiles - global_mean) / global_std
    
    for cluster_id in cluster_zscores.index:
        top_z = cluster_zscores.loc[cluster_id].nlargest(top_n)
        z_str = ", ".join([f"{kmer} (Z={val:.2f})" for kmer, val in top_z.items()])
        print(f"  Klaster {cluster_id} -> {z_str}")
        
    return {
        'seg_crosstab': seg_crosstab,
        'species_crosstab': species_crosstab,
        'cluster_zscores': cluster_zscores,
        'profiles': cluster_profiles,
        'df_analysis': df_analysis
    }


def extract_kmer_summary(res_dict, k_val, dataset_type):
    return {
        'k': k_val,
        'Dataset': dataset_type,
        'Features': 4**k_val,
        'PCA_comps': res_dict['X_pca'].shape[1],
        'KMeans_k': res_dict['params']['kmeans'].get('n_clusters'),
        'KMeans_Sil': round(res_dict['metrics']['kmeans'].get('silhouette', 0), 4),
        'KMeans_DB': round(res_dict['metrics']['kmeans'].get('davies_bouldin', 0), 4),
        'GMM_k': res_dict['params']['gmm'].get('n_components'),
        'GMM_Sil': round(res_dict['metrics']['gmm'].get('silhouette', 0), 4),
        'GMM_DB': round(res_dict['metrics']['gmm'].get('davies_bouldin', 0), 4),
        'Agg_k': res_dict['params']['agglomerative'].get('n_clusters'),
        'Agg_linkage': res_dict['params']['agglomerative'].get('linkage'),
        'Agg_Sil': round(res_dict['metrics']['agglomerative'].get('silhouette', 0), 4),
        'Agg_DB': round(res_dict['metrics']['agglomerative'].get('davies_bouldin', 0), 4)
    }