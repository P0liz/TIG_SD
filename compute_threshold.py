import torch
import numpy as np
import random
import hdbscan
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from torchvision import datasets, transforms
from config import *
from member import Member
from diffusion import pipeline_manager
from mutation_manager import generate
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Local config
NUM_MEMBERS = 500
SEED = 2026
MIN_CLUSTER_SIZE = 1
N_CLUSTERS_RANGE = (5, 30)


# Actually useless unless I want a custom distance
# otherways just use pairwise_distances with the chosen metric
def compute_distance_matrix(members):
    """Compute pairwise euclidean distance matrix between image tensors"""
    n = len(members)
    distance_matrix = np.zeros((n, n))

    print("Computing distance matrix...")
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(members[i].image_tensor - members[j].image_tensor).item()
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    print("Distance matrix sample (first 5x5):")
    print(np.array2string(distance_matrix[:5, :5], precision=2, suppress_small=True))
    print(
        f" min={distance_matrix[distance_matrix > 0].min():.2f}, "
        f" max={distance_matrix.max():.2f}, "
        f" mean={distance_matrix[distance_matrix > 0].mean():.2f}"
    )

    return distance_matrix


def find_optimal_clusters_kmeans(X, n_clusters_range, min_cluster_size=1):
    # KMeans search
    optimal_score = -1
    optimal_n_clusters = -1
    optimal_labels = None

    range_n_clusters = np.arange(n_clusters_range[0], n_clusters_range[1] + 1)

    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)

        cluster_labels = clusterer.fit_predict(X)
        cluster_sizes = np.bincount(cluster_labels)

        if min_cluster_size > 1 and cluster_sizes.min() < min_cluster_size:
            print(f"  k={n_clusters}, SKIPPED (min_size={cluster_sizes.min()} < {min_cluster_size})")
            continue

        silhouette_avg = silhouette_score(X, cluster_labels, metric="euclidean")

        print(f"  k={n_clusters}, silhouette={silhouette_avg:.3f}, sizes={cluster_sizes.tolist()}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters
            optimal_labels = cluster_labels

    if optimal_labels is None:
        raise ValueError("No valid clustering found")

    print(f"Best: k={optimal_n_clusters}, score={optimal_score:.3f}")

    return optimal_labels, optimal_n_clusters


def find_optimal_clusters_AGG(X, n_clusters_range, min_cluster_size=MIN_CLUSTER_SIZE):
    """Find optimal number of clusters using silhouette score"""
    n_samples = X.shape[0]

    max_possible = n_samples // min_cluster_size if min_cluster_size > 1 else n_samples - 1
    upper_bound = min(n_clusters_range[1], max_possible, n_samples - 1)
    range_n_clusters = np.arange(n_clusters_range[0], upper_bound + 1)

    optimal_score = -1
    optimal_n_clusters = -1
    optimal_labels = None

    print("Finding optimal clusters...")
    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="complete")
        cluster_labels = clusterer.fit_predict(X)

        cluster_sizes = np.bincount(cluster_labels)

        if min_cluster_size > 1 and cluster_sizes.min() < min_cluster_size:
            print(f"  n_clusters={n_clusters}, SKIPPED (min_size={cluster_sizes.min()} < {min_cluster_size})")
            continue

        silhouette_avg = silhouette_score(X, cluster_labels, metric="euclidean")
        print(f"  n_clusters={n_clusters}, silhouette={silhouette_avg:.3f}, sizes={cluster_sizes.tolist()}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters
            optimal_labels = cluster_labels

    if optimal_labels is None:
        raise ValueError(f"No valid clustering found with min_cluster_size={min_cluster_size}")

    print(f"Best: n_clusters={optimal_n_clusters}, score={optimal_score:.3f}")
    return optimal_labels, optimal_n_clusters


def agg_cluster_data_thd(X):
    """Cluster using fixed threshold based on distance distribution"""
    distance_matrix = pairwise_distances(X, metric="euclidean")
    # Percentile threshold
    all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    threshold = np.percentile(all_distances, 50)

    # linkage-gap threshold
    # condensed = squareform(distance_matrix)
    # Z = linkage(condensed, method="complete")

    # Largest gap sul dendrogramma
    # merge_distances = Z[:, 2]
    # gaps = np.diff(merge_distances)
    # largest_gap_idx = np.argmax(gaps)
    # threshold = (merge_distances[largest_gap_idx] + merge_distances[largest_gap_idx + 1]) / 2
    # print(f"Largest gap: {gaps[largest_gap_idx]:.3f} → threshold={threshold:.3f}")

    # Trova il gap più grande nell'ultima metà dei merge
    # half_idx = len(merge_distances) // 2
    # gaps = np.diff(merge_distances)
    # largest_gap_idx = np.argmax(gaps)  # + half_idx  # Offset corretto
    # threshold = (merge_distances[largest_gap_idx] + merge_distances[largest_gap_idx + 1]) / 2
    # print(f"Largest gap at merge {largest_gap_idx}: {gaps[np.argmax(gaps)]:.3f} → threshold={threshold:.3f}")

    # Clustering with distance threshold
    print(f"Selected threshold={threshold:.3f}")
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="complete"
    )
    cluster_labels = clusterer.fit_predict(X)

    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = np.bincount(cluster_labels)
    print(f"Found {n_clusters} clusters, sizes: {cluster_sizes}")

    return cluster_labels, n_clusters


def HDBSCAN_clustering(X, min_cluster_size=MIN_CLUSTER_SIZE):
    adjusted_min = 2 if min_cluster_size == 1 else min_cluster_size
    clusterer = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=adjusted_min)
    cluster_labels = clusterer.fit_predict(X)

    if min_cluster_size == 1:
        # Trova il massimo cluster id esistente
        max_label = cluster_labels.max()
        # Trasforma ogni -1 in cluster singolo >> mantieni singoletti
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                max_label += 1
                cluster_labels[i] = max_label
        print(f"HDBSCAN found {len(np.unique(cluster_labels))} clusters (including singletons)")

    return cluster_labels, (
        len(np.unique(cluster_labels)) - 1 if -1 in cluster_labels else len(np.unique(cluster_labels))
    )


def find_optimal_clusters_kmedoids(X_pca, n_clusters_range, min_cluster_size=MIN_CLUSTER_SIZE):
    """Find optimal number of clusters using KMedoids + silhouette score.

    Vantaggi rispetto a KMeans:
    - i medoidi sono punti reali del dataset (utile per interpretabilità)
    - più robusto agli outlier
    - ref: https://scikit-learn-extra.readthedocs.io/en/stable/modules/cluster.html
    """
    n_samples = X_pca.shape[0]
    max_possible = n_samples // min_cluster_size if min_cluster_size > 1 else n_samples - 1
    upper_bound = min(n_clusters_range[1], max_possible, n_samples - 1)
    range_n_clusters = np.arange(n_clusters_range[0], upper_bound + 1)

    # Precomputa distance matrix una volta sola — riusata ad ogni k
    D = pairwise_distances(X_pca, metric="euclidean")

    optimal_score = -1
    optimal_n_clusters = -1
    optimal_labels = None
    optimal_medoid_indices = None

    print("Finding optimal clusters (KMedoids)...")
    for n_clusters in range_n_clusters:
        # method='pam' più accurato ma più lento di 'alternate'
        # init='k-medoids++' analogo a kmeans++, migliore inizializzazione
        cluster_labels, medoid_indices = _kmedoids_fit(D, n_clusters)
        cluster_sizes = np.bincount(cluster_labels)

        if min_cluster_size > 1 and cluster_sizes.min() < min_cluster_size:
            print(f"  k={n_clusters}, SKIPPED (min_size={cluster_sizes.min()} < {min_cluster_size})")
            continue

        silhouette_avg = silhouette_score(X_pca, cluster_labels, metric="euclidean")
        print(f"  k={n_clusters}, silhouette={silhouette_avg:.3f}, sizes={cluster_sizes.tolist()}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters
            optimal_labels = cluster_labels
            optimal_medoid_indices = medoid_indices  # indici reali nel dataset

    if optimal_labels is None:
        raise ValueError(f"No valid clustering found with min_cluster_size={min_cluster_size}")

    print(f"Best: k={optimal_n_clusters}, score={optimal_score:.3f}")
    print(f"Medoid indices: {optimal_medoid_indices.tolist()}")

    return optimal_labels, optimal_n_clusters, optimal_medoid_indices


def _kmedoids_fit(D, n_clusters, random_state=SEED, max_iter=100):
    """PAM KMedoids minimale su distance matrix precomputata"""
    rng = np.random.default_rng(random_state)
    n = D.shape[0]

    medoid_indices = rng.choice(n, n_clusters, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoid_indices], axis=1)

        new_medoids = medoid_indices.copy()
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            if len(cluster_idx) == 0:
                continue
            intra_D = D[np.ix_(cluster_idx, cluster_idx)]
            new_medoids[c] = cluster_idx[np.argmin(intra_D.sum(axis=1))]

        if np.all(new_medoids == medoid_indices):
            break
        medoid_indices = new_medoids

    return labels, medoid_indices


def plot_tsne(X, members, cluster_labels, n_clusters):
    # t-SNE con distance matrix precomputata
    perplexity = max(5, len(members) // 8)  # Adjust perplexity based on sample size
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, init="random", random_state=SEED)
    tsne_results = tsne.fit_transform(X)

    # Palette colori per cluster (condivisa tra i due pannelli)
    cmap = plt.get_cmap("tab20", n_clusters)
    colors = [cmap(cluster_labels[i]) for i in range(len(members))]

    # Marker shapes per label
    marker_list = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8", "x", "1", "2", "3", "4"]
    unique_labels = sorted(set(m.expected_label for m in members))
    label_to_marker = {lbl: marker_list[i % len(marker_list)] for i, lbl in enumerate(unique_labels)}

    # Layout: t-SNE a sinistra (largo), bar chart a destra (stretto)
    fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [7, 1]})

    # ── Pannello sinistro: t-SNE ──────────────────────────────────────────────
    for label in unique_labels:
        marker = label_to_marker[label]
        indices = [i for i, m in enumerate(members) if m.expected_label == label]
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            c=[colors[i] for i in indices],
            marker=marker,
            s=80,
            edgecolors="k",
            linewidths=0.4,
            label=f"label={label}",
            alpha=0.85,
        )

    ax.legend(title="Expected Label", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_title("t-SNE dei image tensors (colore=cluster, forma=label)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # ── Pannello destro: cluster size bar chart ───────────────────────────────
    cluster_ids, cluster_counts = np.unique(cluster_labels, return_counts=True)

    bars = ax_bar.barh(cluster_ids, cluster_counts, color=[cmap(c) for c in cluster_ids], edgecolor="k", linewidth=0.4)

    # Annotazione numerica su ogni barra
    for bar, count in zip(bars, cluster_counts):
        ax_bar.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(count), va="center", ha="left", fontsize=8
        )

    ax_bar.set_yticks(cluster_ids)
    ax_bar.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=8)
    ax_bar.set_xlabel("Membri")
    ax_bar.set_title("Cluster size")
    ax_bar.invert_yaxis()  # cluster 0 in cima, coerente con la legenda
    ax_bar.margins(x=0.2)  # spazio per le annotazioni numeriche

    plt.tight_layout()
    plt.savefig("tsne_clusters.png", dpi=150)
    print("Plot salvato in tsne_clusters.png")
    plt.close()


def compute_cluster_distances_members(X_original, cluster_labels, n_clusters):
    """Compute intra/inter cluster distances using minimum inter-cluster distance"""
    # Precomputa distance matrix (riutilizzabile)
    D = pairwise_distances(X_original, metric="euclidean")

    inter = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            idx_i = np.where(cluster_labels == i)[0]
            idx_j = np.where(cluster_labels == j)[0]
            inter_D = D[np.ix_(idx_i, idx_j)]
            inter.append(inter_D.min())  # Change min to mean or max (single, average, complete)

    return inter


def compute_cluster_distances_medoid(X_original, cluster_labels, n_clusters, medoid_indices):
    # Using medoids instead of centroids because they represent a real element in the dataset
    # and also are more robust to outliers than centroids
    if medoid_indices is None:
        medoid_indices = compute_medoids(X_original, cluster_labels, n_clusters)
    medoids = X_original[medoid_indices]

    inter = [np.linalg.norm(medoids[i] - medoids[j]) for i in range(n_clusters) for j in range(i + 1, n_clusters)]
    return inter


def compute_medoids(X_original, cluster_labels, n_clusters):
    """Calcola medoid per ogni cluster (punto con distanza minima dalla media)"""
    medoid_indices = []

    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        cluster_points = X_original[idx]

        # Medoid = punto con somma distanze minima dagli altri punti del cluster
        D_cluster = pairwise_distances(cluster_points, metric="euclidean")
        medoid_local_idx = np.argmin(D_cluster.sum(axis=1))
        medoid_indices.append(idx[medoid_local_idx])

    return np.array(medoid_indices)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    """ # Use to manually generate memebrs
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Initialize pipeline
    if not pipeline_manager._initialized:
        pipeline_manager.initialize(mode="standard")

    unet_channels = pipeline_manager.pipe.unet.config.in_channels

    # Generate members
    print(f"Generating {NUM_MEMBERS} members...")
    members = []
    prompts = []

    for i in range(NUM_MEMBERS):
        # Random latent
        latent = torch.randn(
            (1, unet_channels, HEIGHT // 8, WIDTH // 8), device=DEVICE, dtype=DTYPE, generator=generator
        )

        # Random prompt and label
        if DATASET == "mnist":
            expected_label = i % len(PROMPTS)  # Cycle through labels for balance
            prompt = PROMPTS[expected_label]
        elif DATASET == "imagenet":
            expected_label = IMAGENET_LABEL
            prompt = random.choice(PROMPTS)

        member = Member(latent, expected_label)
        members.append(member)
        prompts.append(prompt)

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{NUM_MEMBERS} members")

    # Generate images (batch processing)
    print("Generating images...")
    batch_latents = torch.cat([m.latent for m in members], dim=0)
    image_tensors, _ = generate(prompts, batch_latents, generator=generator)

    for member, img_tensor in zip(members, image_tensors):
        member.image_tensor = img_tensor
    """

    # Load MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Create members from MNIST
    print(f"Loading {NUM_MEMBERS} images from MNIST test set...")
    members = []

    for i in range(min(NUM_MEMBERS, len(mnist_test))):
        image_tensor, label = mnist_test[i]

        # Create member with dummy latent (not used for MNIST clustering)
        latent = torch.zeros((1, 4, HEIGHT // 8, WIDTH // 8), device=DEVICE, dtype=DTYPE)
        member = Member(latent, expected_label=label)
        member.image_tensor = image_tensor

        members.append(member)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{NUM_MEMBERS} images")

    print(f"Loaded {len(members)} members from MNIST")

    # Build tensor feature matrix (N x D)
    image_tensors = np.stack([m.image_tensor.detach().cpu().numpy() for m in members])
    # flatten and normalize
    if DATASET == "mnist":
        X_flat = image_tensors.reshape(len(members), -1)
    elif DATASET == "imagenet":
        X_flat = image_tensors.reshape(len(members), -1) / 255.0
    print("Feature matrix values:")
    print(f"shape: {X_flat.shape}")
    print(f"dtype: {X_flat.dtype}")
    print(f"min={X_flat.min():.3f}, max={X_flat.max():.3f}")

    # PCA
    # get number of components comprising 95% of the variance
    # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
    # pca = PCA(n_components=0.95, random_state=SEED)
    # X_pca = pca.fit_transform(X_flat)
    # print(f"PCA reduced shape: {X_flat.shape} → {X_pca.shape}")
    # print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Compute custom distance matrix
    # distance_matrix = compute_distance_matrix(members)

    # Clustering
    medoid_indices = None
    cluster_labels, n_clusters, medoid_indices = find_optimal_clusters_kmedoids(X_flat, N_CLUSTERS_RANGE)
    # cluster_labels, n_clusters = find_optimal_clusters_kmeans(X_flat, N_CLUSTERS_RANGE)
    # cluster_labels, n_clusters = find_optimal_clusters_AGG(X_flat, N_CLUSTERS_RANGE)
    # cluster_labels, n_clusters = agg_cluster_data_thd(X_flat)
    # cluster_labels, n_clusters = HDBSCAN_clustering(X_flat)

    # Distances Debug
    """
    print(f"\nDEBUG INFO:")
    print(f"Medoid indices: {medoid_indices}")
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")

    # Verifica distribuzione distanze
    all_dists = pairwise_distances(X_flat, metric="euclidean")
    print(
        f"All pairwise distances: min={all_dists[all_dists>0].min():.3f}, "
        f"max={all_dists.max():.3f}, mean={all_dists[all_dists>0].mean():.3f}"
    )

    # Distanze tra medoids
    if medoid_indices is not None:
        medoid_dists = pairwise_distances(X_flat[medoid_indices], metric="euclidean")
        print(f"Medoid distances: min={medoid_dists[medoid_dists>0].min():.3f}, " f"max={medoid_dists.max():.3f}")
    """

    # Plotting
    plot_tsne(X_flat, members, cluster_labels, n_clusters)

    # Compute inter-cluster distances
    inter_cluster_dists = compute_cluster_distances_medoid(X_flat, cluster_labels, n_clusters, medoid_indices)
    # inter_cluster_dists = compute_cluster_distances_members(X_flat, cluster_labels, n_clusters)

    # Results
    avg_inter_cluster_dist = np.mean(inter_cluster_dists)
    std_inter_cluster_dist = np.std(inter_cluster_dists)
    median_inter_cluster_dist = np.median(inter_cluster_dists)

    print("\n=== RESULTS ===")
    print(f"Total members: {NUM_MEMBERS}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average inter-cluster distance: {avg_inter_cluster_dist:.3f}")
    print(f"Std inter-cluster distance: {std_inter_cluster_dist:.3f}")
    print(f"Median inter-cluster distance: {median_inter_cluster_dist:.3f}")
    print(f"Min inter-cluster distance: {min(inter_cluster_dists):.3f}")
    print(f"Max inter-cluster distance: {max(inter_cluster_dists):.3f}")
    print(f"Recommended DIST_THRESHOLD: {avg_inter_cluster_dist:.3f}\n\n")

    return avg_inter_cluster_dist, std_inter_cluster_dist, n_clusters


if __name__ == "__main__":
    import csv

    results = []
    for i in range(1, 11):
        NUM_MEMBERS = i * 100
        avg, std, n_clusters = main()
        results.append([NUM_MEMBERS, n_clusters, avg, std])

    # Calcola media delle medie
    avg_of_avgs = np.mean([r[2] for r in results])
    avg_of_stds = np.mean([r[3] for r in results])

    with open("distances.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["members", "n_clusters", "avg_dist", "std_dist"])
        writer.writerows(results)
        writer.writerow(["MEAN", "-", f"{avg_of_avgs:.3f}", f"{avg_of_stds:.3f}"])

    print("\n=== OVERALL SUMMARY ===")
    print(f"Mean of avg distances: {avg_of_avgs:.3f}")
    print(f"Mean of std distances: {avg_of_stds:.3f}")
    print("Results saved to distances.csv")
