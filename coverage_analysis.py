# %%writefile coverage_analysis.py

import numpy as np
import glob
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === CONFIG ===
SEED = 7  # For reproducibility
METHODS_TO_COMPARE = {
    "archive_size": "runs/archive_size/run_*/",
    "archive_dist": "runs/archive_dist/run_*/",
    "archive_bucket": "runs/archive_bucket/run_*/",
}

""" Assure the dataset of runs has the following structure
runs/archive_size/run_*/
    inds/
        ind20/
            data.json
            m1_latent.npy
            m2_latent.npy
            ...
        ind35/
            ...
"""

OUTPUT_FOLDER = "diversity_analysis_results"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)


# === FUNCTIONS ===
def cluster_data(data, n_clusters_interval):
    """Trova il numero ottimale di cluster usando silhouette score"""
    assert n_clusters_interval[0] >= 2, "Min number of clusters must be >= 2"
    range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
    optimal_score = -1
    optimal_n_clusters = -1

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"  n_clusters={n_clusters}, silhouette={silhouette_avg:.3f}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters

    assert optimal_n_clusters != -1, "Error in silhouette analysis"
    print(f"Best: n_clusters={optimal_n_clusters}, score={optimal_score:.3f}")

    clusterer = KMeans(n_clusters=optimal_n_clusters, random_state=SEED).fit(data)
    return clusterer.labels_, clusterer.cluster_centers_


# Using as main features the concatenation of the members latent vectors
def load_archived_individuals(folder_pattern, method_name):
    """Load individuals from inds folder"""
    all_individuals = []
    run_folders = glob.glob(folder_pattern)

    print(f"Loading {method_name}: found {len(run_folders)} runs")

    for run_folder in run_folders:
        inds_folder = Path(run_folder) / "inds"

        if not inds_folder.exists():
            continue

        for ind_dir in inds_folder.glob("ind*"):
            json_file = ind_dir / "data.json"
            m1_latent_file = ind_dir / "m1_latent.npy"
            m2_latent_file = ind_dir / "m2_latent.npy"

            if not json_file.exists() or not m1_latent_file.exists() or not m2_latent_file.exists():
                continue

            with open(json_file, "r") as f:
                ind_data = json.load(f)

            # Load both latents
            m1_latent = np.load(m1_latent_file).flatten()
            m2_latent = np.load(m2_latent_file).flatten()

            # Compute mean and distance (maybe add as features)
            # mean_latent = (m1_latent + m2_latent) / 2.0
            # distance = ind_data["members_distance"]

            # Individual representation: concatenated [m1_latent, m2_latent]
            # It just make sense because k-means uses euclidean distance
            # so similarity is seen also between single members
            individual_vector = np.concatenate([m1_latent, m2_latent])

            # Append to list of all individuals, categorized by method
            # in case add other data to use later
            all_individuals.append([individual_vector, method_name])

    print(f"  Loaded {len(all_individuals)} individuals")
    return all_individuals


def compute_coverage_and_plot(all_data, method_names):
    """Calcola coverage e visualizza con t-SNE"""

    # 4 is a reasonable minimum for stable clustering with silhouette scoring
    if len(all_data) < 4:
        print("Not enough data for clustering (need at least 4 samples)")
        return

    # Prepare data
    # X: Matrix of shape (n_samples, 2*latent_dim) - concatenated member latents
    X = np.array([sample[0] for sample in all_data])
    labels_method = [sample[1] for sample in all_data]

    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")

    # Clustering
    max_clusters = min(len(X) - 1, 20)
    cluster_labels, centers = cluster_data(X, (2, max_clusters))
    num_clusters = len(centers)

    print(f"Total clusters: {num_clusters}")

    # Coverage calculation for each method
    coverage_dict = {}
    for method in method_names:
        method_indices = [i for i, label in enumerate(labels_method) if label == method]
        method_clusters = set(cluster_labels[i] for i in method_indices)
        coverage = len(method_clusters) / num_clusters * 100  # with percentage
        coverage_dict[method] = coverage

        print(
            f"{method}: {len(method_indices)} samples, "
            f"coverage={coverage:.2f}% ({len(method_clusters)}/{num_clusters} clusters)"
        )

    # t-SNE visualization
    print("Computing t-SNE...")
    # Combine data and centroids for single t-SNE transformation
    X_with_centers = np.vstack([X, centers])
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(30, len(X) - 1), n_iter=3000, random_state=SEED)
    tsne_results_all = tsne.fit_transform(X_with_centers)

    # Split back into data points and centroids
    tsne_results = tsne_results_all[: len(X)]  # before len(X)
    centers_tsne = tsne_results_all[len(X) :]  # after len(X)

    # Plot
    plt.figure(figsize=(12, 8))

    colors = ["blue", "green", "red"]
    markers = ["s", "d", "b"]

    for i, method in enumerate(method_names):
        method_indices = [j for j, label in enumerate(labels_method) if label == method]
        plt.scatter(
            tsne_results[method_indices, 0],
            tsne_results[method_indices, 1],
            c=colors[i],
            marker=markers[i],
            label=f"{method} ({coverage_dict[method]:.1f}%)",
            alpha=0.6,
            s=80,
        )

    # Plot centroids
    plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], c="yellow", marker="x", s=200, linewidths=3, label="Centroids")

    plt.legend()
    plt.title(f"Archive Diversity Comparison (t-SNE)\nTotal clusters: {num_clusters}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/diversity_comparison.pdf")
    print(f"Plot saved to {OUTPUT_FOLDER}/diversity_comparison.pdf")

    return coverage_dict, num_clusters


# TODO: add statistical analysis
# === MAIN ===
if __name__ == "__main__":
    all_data = []  # Store individuals data
    method_names = []  # Store corresponding method used

    # Load data from all methods
    for method_name, folder_pattern in METHODS_TO_COMPARE.items():
        individuals = load_archived_individuals(folder_pattern, method_name)
        all_data.extend(individuals)
        if len(individuals) > 0:
            method_names.append(method_name)

    if len(all_data) == 0:
        print("ERROR: No data loaded!")
        exit(1)

    # Analisi
    coverage_dict, num_clusters = compute_coverage_and_plot(all_data, method_names)

    # Salva risultati
    results = {"num_clusters": num_clusters, "coverage": coverage_dict, "total_samples": len(all_data)}

    with open(f"{OUTPUT_FOLDER}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(f"Total clusters: {num_clusters}")
    for method, cov in coverage_dict.items():
        print(f"{method}: {cov:.2f}% coverage")

    print(f"\nResults saved to {OUTPUT_FOLDER}/")

# TODO: test clustering with distance between individuals
# and review the code below
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def compute_individual_distance_matrix(all_data):
    """Compute pairwise distance matrix using Individual.distance()"""
    n = len(all_data)
    distance_matrix = np.zeros((n, n))

    print("Computing custom distance matrix...")
    for i in range(n):
        for j in range(i + 1, n):
            # Load individuals and compute custom distance
            ind_i = all_data[i]  # Contains [individual_vector, method_name]
            ind_j = all_data[j]

            # Extract m1 and m2 latents from concatenated vector
            latent_dim = len(ind_i[0]) // 2
            m1_i = ind_i[0][:latent_dim]
            m2_i = ind_i[0][latent_dim:]
            m1_j = ind_j[0][:latent_dim]
            m2_j = ind_j[0][latent_dim:]

            # Compute distance formula
            a = np.linalg.norm(m1_i - m1_j)  # i1.m1 vs i2.m1
            b = np.linalg.norm(m1_i - m2_j)  # i1.m1 vs i2.m2
            c = np.linalg.norm(m2_i - m1_j)  # i1.m2 vs i2.m1
            d = np.linalg.norm(m2_i - m2_j)  # i1.m2 vs i2.m2
            dist = np.mean([min(a, b), min(c, d), min(a, c), min(b, d)])

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def agg_cluster_data(data, n_clusters_interval):
    """Cluster using precomputed custom distance matrix"""
    assert n_clusters_interval[0] >= 2, "Min number of clusters must be >= 2"

    # Compute custom distance matrix
    distance_matrix = compute_individual_distance_matrix(data)

    range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
    optimal_score = -1
    optimal_n_clusters = -1

    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        cluster_labels = clusterer.fit_predict(distance_matrix)
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        print(f"  n_clusters={n_clusters}, silhouette={silhouette_avg:.3f}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters

    assert optimal_n_clusters != -1, "Error in silhouette analysis"
    print(f"Best: n_clusters={optimal_n_clusters}, score={optimal_score:.3f}")

    clusterer = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric="precomputed", linkage="average")
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Compute centers (medoids - closest point to cluster mean)
    centers = []
    for i in range(optimal_n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_idx = cluster_indices[cluster_distances.sum(axis=1).argmin()]
        centers.append(data[medoid_idx][0])  # Get the individual vector

    return cluster_labels, np.array(centers)
