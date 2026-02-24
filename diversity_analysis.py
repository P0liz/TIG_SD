import numpy as np
import glob
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm

from data_visualization import plot_coverage_venn
from config import ANALYSIS_CONFIG, DIVERSITY_OUTPUT_FOLDER, FOCUS_NAME, OTHERS_NAME, PROMPTS

# === LOCAL CONFIG ===
SEED = 7  # For reproducibility
MIN_CLUSTER_SIZE = 1  # Min samples per cluster
PERCENTILE_THD = 25  # For clustering threshold

if ANALYSIS_CONFIG == "single_run":
    METHODS_TO_COMPARE = {
        FOCUS_NAME: "runs/archive_bucket/run_123",  # specific run
        OTHERS_NAME: [  # Include tutte le altre run
            "runs/archive_size/run_*/",
            "runs/archive_dist/run_*/",
            "runs/archive_bucket_size/run_*/",
            "runs/archive_bucket_dist/run_*/",
        ],
    }
elif ANALYSIS_CONFIG == "archives":
    METHODS_TO_COMPARE = {
        "archive_size": "runs/archive_size/run_*/",
        "archive_dist": "runs/archive_dist/run_*/",
        "archive_bucket_size": "runs/archive_bucket_size/run_*/",
        "archive_bucket_dist": "runs/archive_bucket_dist/run_*/",
    }

""" Make sure the dataset of runs has the following structure
runs/archive_size/run_*/
    inds/
        ind20/
            data.json
            m1_latent.npy
            m2_latent.npy
            ...
        ind*/
            ...
"""


# =================================================================
# FUNCTIONS
# =================================================================
def load_archived_individuals(folder_pattern, method_name, exclude_path=None):
    all_individuals = []

    if isinstance(folder_pattern, str):
        folder_patterns = [folder_pattern]
    else:
        folder_patterns = folder_pattern

    # Raccogli tutte le run folders
    all_run_folders = []
    for pattern in folder_patterns:
        run_folders = glob.glob(pattern)
        all_run_folders.extend(run_folders)
    # Rimuovi duplicati e ordina
    all_run_folders = sorted(set(all_run_folders))

    # If specified exclude a certain run
    if exclude_path:
        all_run_folders = [f for f in all_run_folders if f != exclude_path]
        print(f"Loading {method_name}: found {len(all_run_folders)} runs " f"(excluded {Path(exclude_path).name})")
    else:
        print(f"Loading {method_name}: found {len(all_run_folders)} runs")

    inds_per_run = []
    elapsed_times = []  # in secondi
    for run_folder in all_run_folders:
        inds_folder = Path(run_folder) / "inds"

        if not inds_folder.exists():
            print(f"Warning: inds folder not found in {run_folder}")
            continue

        # Leggi elapsed time da stats.csv se presente
        stats_file = Path(run_folder) / "stats.csv"
        if stats_file.exists():
            try:
                df = pd.read_csv(stats_file, skipinitialspace=True)
                df.columns = df.columns.str.strip()  # rimuove spazi dai nomi colonne
                df["iteration"] = df["iteration"].str.strip()  # rimuove spazi dai valori

                final_row = df[df["iteration"] == "final"]
                if not final_row.empty:
                    time_str = final_row["elapsed_time"].values[0].strip()
                    td = pd.to_timedelta(time_str)
                    elapsed_times.append(td.total_seconds())
            except Exception as e:
                print(f"Warning: could not read stats.csv in {run_folder}: {e}")

        inds_count = 0
        for ind_dir in inds_folder.glob("ind*"):
            json_file = ind_dir / "data.json"
            m1_latent_file = ind_dir / "m1_latent.npy"
            m2_latent_file = ind_dir / "m2_latent.npy"

            if not (json_file.exists() and m1_latent_file.exists() and m2_latent_file.exists()):
                continue

            with open(json_file, "r") as f:
                ind_data = json.load(f)

            # Load data for this individual
            m1_latent = np.load(m1_latent_file)
            m2_latent = np.load(m2_latent_file)
            individual_data = (m1_latent, m2_latent)
            expected_label = ind_data.get("m1", {}).get("expected_label", "unknown")

            # Append to list
            all_individuals.append([method_name, individual_data, expected_label])
            inds_count += 1

        inds_per_run.append(inds_count)

    print(f"Loaded {len(all_individuals)} individuals")
    # Get individuals per run stats
    avg_inds = np.mean(inds_per_run)
    std_inds = np.std(inds_per_run)
    avg_elapsed = np.mean(elapsed_times)
    return all_individuals, avg_inds, std_inds, avg_elapsed


def compute_individual_distance_matrix(inds_data):
    """Compute pairwise distance matrix using Individual distance"""
    n = len(inds_data)
    distance_matrix = np.zeros((n, n))

    print("Computing custom distance matrix...")
    for i in range(n):
        for j in range(i + 1, n):
            # Load individuals as couples
            m1_i, m2_i = inds_data[i][1]
            m1_j, m2_j = inds_data[j][1]

            # Compute distance formula
            a = np.linalg.norm(m1_i - m1_j)  # i1.m1 vs i2.m1
            b = np.linalg.norm(m1_i - m2_j)  # i1.m1 vs i2.m2
            c = np.linalg.norm(m2_i - m1_j)  # i1.m2 vs i2.m1
            d = np.linalg.norm(m2_i - m2_j)  # i1.m2 vs i2.m2
            dist = np.mean([min(a, b), min(c, d), min(a, c), min(b, d)])

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


# === CLUSTERING ===
def agg_cluster_data_silh(inds_data, n_clusters_interval, distance_matrix, min_cluster_size=MIN_CLUSTER_SIZE):
    assert n_clusters_interval[0] >= 2, "Min number of clusters must be >= 2"

    n_samples = len(inds_data)

    # Apply min_cluster_size constraint
    if min_cluster_size > 1:
        max_possible = n_samples // min_cluster_size
        upper_bound = min(n_clusters_interval[1], max_possible, n_samples - 1)
        print(f"Sample size: {n_samples}, min_cluster_size: {min_cluster_size}")
        print(f"Max possible clusters: {max_possible}")
    else:
        upper_bound = min(n_clusters_interval[1], n_samples - 1)

    range_n_clusters = np.arange(n_clusters_interval[0], upper_bound + 1)
    optimal_score = -1
    optimal_n_clusters = -1
    optimal_labels = None

    # Silhouette analysis to determine the number of clusters
    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        cluster_labels = clusterer.fit_predict(distance_matrix)

        # Check cluster sizes
        cluster_sizes = np.bincount(cluster_labels)

        # Skip if min_cluster_size violated
        if min_cluster_size is not None and cluster_sizes.min() < min_cluster_size:
            print(f"  n_clusters={n_clusters}, SKIPPED (min_size={cluster_sizes.min()} < {min_cluster_size})")
            continue

        # Compute silhouette
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")

        print(f"  n_clusters={n_clusters}, silhouette={silhouette_avg:.3f}, " f"sizes={cluster_sizes.tolist()}")

        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters
            optimal_labels = cluster_labels

    assert optimal_n_clusters != -1, "Error in silhouette analysis"
    print(f"Best: n_clusters={optimal_n_clusters}, score={optimal_score:.3f}")
    cluster_sizes = np.bincount(optimal_labels)
    print(f"Final cluster sizes: {cluster_sizes}")

    # Compute medoids (real points)
    centers = []
    for k in range(optimal_n_clusters):
        cluster_indices = np.where(optimal_labels == k)[0]

        # Medoid = point with minimum total distance to others in cluster
        cluster_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_local_idx = cluster_dist.sum(axis=1).argmin()
        medoid_idx = cluster_indices[medoid_local_idx]

        centers.append(inds_data[medoid_idx][1])  # Individual vectors couple

    return optimal_labels, np.array(centers)


def agg_cluster_data_thd(inds_data, distance_matrix):
    """Cluster using fixed threshold based on distance distribution"""
    all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    threshold = np.percentile(all_distances, PERCENTILE_THD)
    print(f"Selected threshold={threshold:.3f}")

    # Clustering with distance threshold
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="precomputed", linkage="average"
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = np.bincount(cluster_labels)
    print(f"Found {n_clusters} clusters, sizes: {cluster_sizes}")

    # Compute medoids
    centers = []
    for k in range(n_clusters):
        cluster_indices = np.where(cluster_labels == k)[0]
        cluster_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_idx = cluster_indices[cluster_dist.sum(axis=1).argmin()]
        centers.append(inds_data[medoid_idx][1])  # Individual vectors couple

    return cluster_labels, np.array(centers)


# === COVERAGE CALCULATION ===
def compute_coverage_and_plot(inds_data, method_names, distance_matrix, idx=0):
    """Calcola coverage e visualizza con t-SNE"""

    if len(inds_data) < 4:
        print("Not enough data for clustering (need at least 4 samples)")
        return

    # Metodo a cui ogni individuo appartiene
    labels_method = [sample[0] for sample in inds_data]
    print(f"Total samples: {len(inds_data)}")

    # Clustering con range per numero di cluster
    # max_clusters = min(len(inds_data) - 1, 20)
    # cluster_labels, centers = agg_cluster_data_silh(inds_data, (2, max_clusters), distance_matrix)

    # Clustering con threshold fisso per taglio dendogramma
    cluster_labels, centers = agg_cluster_data_thd(inds_data, distance_matrix)

    num_clusters = len(centers)
    print(f"Total clusters: {num_clusters}")

    # Coverage calculation between archives
    if ANALYSIS_CONFIG == "archives":
        # Calcolo coverage semplice (unweighted) per ogni archivio
        # coverage_dict_weighted = compute_archives_coverage(labels_method, cluster_labels, method_names)

        # Calcolo coverage pesato
        coverage_dict_weighted, coverage_dict_unweighted = compute_weighted_coverage(
            labels_method, cluster_labels, method_names, distance_matrix
        )

        plot_coverage_venn(inds_data, cluster_labels, method_names)

    # Coverage run focus vs tutte le altre
    elif ANALYSIS_CONFIG == "single_run":
        # Calcolo coverage semplice per la run focus rispetto a tutte le altre
        # coverage_dict_weighted, _ = compute_run_vs_others_coverage(labels_method, cluster_labels, method_names)

        # Calcolo coverage pesato
        coverage_dict_weighted, coverage_dict_unweighted = compute_weighted_coverage(
            labels_method, cluster_labels, method_names, distance_matrix
        )

    # Visualizzazione t-SNE usando la matrice di distanza personalizzata
    print(f"Feature dimension for t-SNE: {distance_matrix.shape[1]}")
    perplexity = max(5, len(inds_data) // 8)  # Adjust perplexity based on sample size
    tsne_results = TSNE(
        n_components=2, verbose=1, perplexity=perplexity, metric="precomputed", init="random", random_state=SEED
    ).fit_transform(distance_matrix)

    # Calcola centroids NELLO SPAZIO t-SNE (2D)
    centers_tsne = []
    for cluster_id in range(len(centers)):
        cluster_mask = cluster_labels == cluster_id
        cluster_points_tsne = tsne_results[cluster_mask]

        if len(cluster_points_tsne) > 0:
            # Media dei punti del cluster NELLO SPAZIO t-SNE
            center_tsne = cluster_points_tsne.mean(axis=0)
            centers_tsne.append(center_tsne)
        else:
            centers_tsne.append([0, 0])  # Fallback se cluster vuoto
    centers_tsne = np.array(centers_tsne)

    # Plot
    plt.figure(figsize=(14, 10))

    # Genera colori per cluster
    cluster_colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    # Marker per metodo
    if ANALYSIS_CONFIG == "single_run":
        method_markers = {FOCUS_NAME: "v", OTHERS_NAME: "o"}
    elif ANALYSIS_CONFIG == "archives":
        method_markers = {
            "archive_size": "s",
            "archive_dist": "d",
            "archive_bucket_size": "^",
            "archive_bucket_dist": "o",
        }
    else:
        raise ValueError(f"Unknown ANALYSIS_CONFIG: {ANALYSIS_CONFIG}")

    # Plot centroids con colori corrispondenti ai cluster
    plt.scatter(
        centers_tsne[:, 0],
        centers_tsne[:, 1],
        c=cluster_colors,
        marker="x",
        s=150,
        alpha=0.6,
        linewidths=4,
        zorder=0,
        label="Centroids",
    )

    # Plot per cluster e metodo
    plotted_methods = set()  # Track quali metodi sono già in legend
    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_color = cluster_colors[cluster_id]

        for method in method_names:
            # Punti che sono sia nel cluster che del metodo
            method_and_cluster_mask = [cluster_mask[i] and labels_method[i] == method for i in range(len(inds_data))]

            if any(method_and_cluster_mask):
                # Label solo la prima volta per ogni metodo
                label = f"{method} ({coverage_dict_weighted[method]:.1f}%)" if method not in plotted_methods else None
                if label:
                    plotted_methods.add(method)

                # Increase size to make focus run more visible
                size = 400 if ANALYSIS_CONFIG == "single_run" and method == "focus" else 80

                plt.scatter(
                    tsne_results[method_and_cluster_mask, 0],
                    tsne_results[method_and_cluster_mask, 1],
                    c=[cluster_color],
                    marker=method_markers[method],
                    s=size,
                    zorder=2,
                    label=label,
                )

    plt.legend()
    if ANALYSIS_CONFIG == "single_run":
        plt.title(f"Single Run Diversity Comparison (t-SNE) Total clusters: {num_clusters}")
    else:
        plt.title(f"Archive Diversity Comparison (t-SNE) Total clusters: {num_clusters}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    plt.tight_layout()
    plt.savefig(f"{DIVERSITY_OUTPUT_FOLDER}/diversity_comparison_{idx}.pdf")

    return coverage_dict_weighted, coverage_dict_unweighted, num_clusters


def compute_archives_coverage(labels_method, cluster_labels, method_names):
    """Calcola coverage di ogni archivio rispetto al totale dei cluster"""
    assert ANALYSIS_CONFIG == "archives", "This function is only for archives mode"
    num_clusters = len(set(cluster_labels))

    coverage_dict = {}
    for method in method_names:
        method_indices = [i for i, label in enumerate(labels_method) if label == method]
        method_clusters = set(cluster_labels[i] for i in method_indices)
        coverage = len(method_clusters) / num_clusters * 100 if num_clusters > 0 else 0
        coverage_dict[method] = coverage
        print(
            f"{method}: {len(method_indices)} samples, "
            f"coverage={coverage:.2f}% ({len(method_clusters)}/{num_clusters} clusters)"
        )
    return coverage_dict


def compute_single_runs_coverage(labels_method, cluster_labels, method_names):
    """Calcola il coverage di ogni singola run rispetto a tutte le altre"""
    assert ANALYSIS_CONFIG == "single_run", "This function is only for single_run mode"
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    coverage_dict = {}
    for method in method_names:
        focus_indices = [i for i, m in enumerate(labels_method) if m == method]
        focus_clusters = set(cluster_labels[i] for i in focus_indices if cluster_labels[i] != -1)
        coverage = len(focus_clusters) / num_clusters * 100 if num_clusters > 0 else 0
        coverage_dict[method] = coverage
        print(f"Coverage di '{method}': {len(focus_clusters)}/{num_clusters} cluster " f"({coverage:.2f}%)")
    return coverage_dict, focus_clusters


def compute_weighted_coverage(labels_method, cluster_labels, method_names, distance_matrix):
    """
    Calcola coverage pesata considerando:
    1. Numero di cluster coperti (quanto è "sparso" nel solution space)
    2. Diversità interna degli individui di quel metodo (penalizza tanti individui troppo simili tra loro)
    """
    num_clusters = len(set(cluster_labels))

    print("Coverage calculation with weighting factors...")
    coverage_weighted = {}
    coverage_unweighted = {}

    for method in method_names:
        method_indices = [i for i, label in enumerate(labels_method) if label == method]
        num_individuals = len(method_indices)

        # Cluster coperti da questo metodo
        method_clusters = {}
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            covered_in_cluster = [i for i in method_indices if cluster_mask[i]]
            if covered_in_cluster:
                method_clusters[cluster_id] = len(covered_in_cluster)

        num_clusters_covered = len(method_clusters)

        # Coverage semplice (unweighted)
        coverage_simple = (num_clusters_covered / num_clusters * 100) if num_clusters > 0 else 0
        coverage_unweighted[method] = coverage_simple

        # Calcola diversità INTERNA degli individui del metodo
        method_diversity = 0.0
        if len(method_indices) > 1:
            method_dist_matrix = distance_matrix[np.ix_(method_indices, method_indices)]
            mask = ~np.eye(len(method_indices), dtype=bool)  # Esclude diagonale (distanza con se stessi)
            method_distances = method_dist_matrix[mask]
            if len(method_distances) > 0:
                avg_method_dist = np.mean(method_distances)
                max_dist = np.max(distance_matrix)
                method_diversity = avg_method_dist / max_dist if max_dist > 0 else 0.5
                method_diversity = min(method_diversity, 1.0)

        # Coverage pesata: combina fattori
        # 1. Coverage semplice (numero cluster coperti)
        # 2. Diversità interna (quanto sono diversi gli individui del metodo)
        coverage_weighted_final = (
            (coverage_simple / 100) * 0.70  # 70% peso: quanti cluster copre
            + method_diversity * 0.30  # 30% peso: quanto sono diversi internamente
        ) * 100

        coverage_weighted[method] = coverage_weighted_final

        print(f"\n{method}:")
        print(f"  Individuals: {num_individuals}")
        print(f"  Clusters covered: {num_clusters_covered}/{num_clusters}")
        print(f"  Coverage (unweighted): {coverage_simple:.2f}%")
        print(f"  Internal diversity: {method_diversity:.3f}")
        print(f"  Coverage (weighted/final): {coverage_weighted_final:.2f}%")

    return coverage_weighted, coverage_unweighted


# === SINGLE RUN ANALYSIS HELPERS ===
def compute_all_single_runs():
    """
    Calcola coverage di OGNI run rispetto a TUTTE le altre.
    Restituisce un dizionario: {run_path: coverage}
    """
    # Carica TUTTE le run di TUTTI gli archivi
    folder_patterns = METHODS_TO_COMPARE[OTHERS_NAME]

    all_run_folders = []
    for pattern in folder_patterns:
        all_run_folders.extend(glob.glob(pattern))
    # Rimuovi duplicati e ordina
    all_run_folders = sorted(set(all_run_folders))

    # Remove runs with no archived individuals
    for folder in all_run_folders:
        inds_folder = Path(folder) / "inds"
        if not inds_folder.exists():
            all_run_folders.remove(folder)
    print(f"Total runs found: {len(all_run_folders)}")

    # Load all inds data from all runs (for global distance matrix)
    all_individuals_data = []
    for folder in all_run_folders:
        individuals, _, _, _ = load_archived_individuals(folder, folder)  # using folder as method_name just for loading
        all_individuals_data.extend(individuals)

    # Calculate global distance matrix once for all individuals (efficiency)
    global_distance_matrix = compute_individual_distance_matrix(all_individuals_data)
    global_paths = [sample[0] for sample in all_individuals_data]  # label = run path

    coverage_results = {}  # {run_path: coverage_percentage}
    for idx, focus_run in enumerate(all_run_folders):
        # Start index from 1 for better readability
        # Index 0 reserved for full archive all runs analysis if needed
        idx = idx + 1
        print(f"\n=== PROCESSING RUN {idx}/{len(all_run_folders)}: {Path(focus_run).name} ===")

        """
        # Prepara config per questa iterazione
        # Focus: questa run
        # Others: TUTTE le run (di tutti gli archivi)
        temp_methods = {FOCUS_NAME: focus_run, OTHERS_NAME: all_run_folders}

        # Carica dati
        inds_data = []
        method_names = []

        for method_name, folder_pattern in temp_methods.items():
            # Exclude focus run from "others"
            exclude_path = focus_run if method_name != "focus" else None
            individuals, _, _, _ = load_archived_individuals(folder_pattern, method_name, exclude_path)
            inds_data.extend(individuals)
            if len(individuals) > 0:
                method_names.append(method_name)

        # Compute custom distance matrix
        distance_matrix = compute_individual_distance_matrix(inds_data)
        """
        # Trova gli indici nella matrice globale
        focus_indices = [i for i, p in enumerate(global_paths) if p == focus_run]
        others_indices = [i for i, p in enumerate(global_paths) if p != focus_run]

        all_indices = focus_indices + others_indices

        # Estrai sottomatrice già calcolata
        sub_matrix = global_distance_matrix[np.ix_(all_indices, all_indices)]

        # Ricostruisci inds_data con le label corrette per questa iterazione
        groups = {FOCUS_NAME: focus_indices, OTHERS_NAME: others_indices}
        inds_data = [[label, *all_individuals_data[i][1:]] for label, indices in groups.items() for i in indices]

        if len(inds_data) < 4:
            print(f"Not enough data, skipping")
            continue

        print(f"Total samples: {len(inds_data)}")
        print(f"Focus: {len([d for d in inds_data if d[0]=='focus'])} samples")
        print(f"Others: {len([d for d in inds_data if d[0]=='others'])} samples")

        # Calcola coverage
        try:
            # Analisi
            coverage_dict_w, coverage_dict_uw, _ = compute_coverage_and_plot(
                inds_data, [FOCUS_NAME, OTHERS_NAME], sub_matrix, idx
            )
            coverage_results[focus_run] = (coverage_dict_w[FOCUS_NAME], coverage_dict_uw[FOCUS_NAME])

        except Exception as e:
            print(f"Error: {e}")
            continue

    return coverage_results


def group_coverages_by_archive(coverage_results):
    """
    Raggruppa le coverage per archivio di appartenenza.
    """
    grouped = {"archive_size": [], "archive_dist": [], "archive_bucket_size": [], "archive_bucket_dist": []}

    for run_path, coverage in coverage_results.items():
        # coverage è una tupla (weighted, unweighted)
        # Identifica archivio dalla path
        if "archive_size" in run_path:
            grouped["archive_size"].append(coverage)
        elif "archive_dist" in run_path:
            grouped["archive_dist"].append(coverage)
        elif "archive_bucket_size" in run_path:
            grouped["archive_bucket_size"].append(coverage)
        elif "archive_bucket_dist" in run_path:
            grouped["archive_bucket_dist"].append(coverage)
    return grouped


def calculate_label_coverage(inds_data):
    covered_labels = {}  # {expected_label: count}
    for sample in inds_data:
        label = sample[2]  # The expected label
        covered_labels[label] = covered_labels.get(label, 0) + 1
    label_coverage = len(covered_labels) / len(PROMPTS) * 100
    return label_coverage


# =================================================================
# MAINS
# =================================================================
def single_run_main():
    print(f"COMPUTING COVERAGE FOR ALL RUNS")
    print(f"Each run vs all other runs (across all archives)")

    # Calcola coverage di tutte le run (una volta sola)
    coverage_results = compute_all_single_runs()

    # Raggruppa per archivio
    grouped_coverages = group_coverages_by_archive(coverage_results)

    # Calcola statistiche per archivio
    print(f"RESULTS GROUPED BY ARCHIVE")

    all_results = {}
    results_list = []

    for archive_name in ["archive_size", "archive_dist", "archive_bucket_size", "archive_bucket_dist"]:
        both_coverages = grouped_coverages[archive_name]

        # Separa weighted e unweighted
        coverages_weighted = [cov[0] for cov in both_coverages] if both_coverages else []
        coverages_unweighted = [cov[1] for cov in both_coverages] if both_coverages else []

        # Statistiche weighted
        if coverages_weighted:
            avg_w = float(np.mean(coverages_weighted))
            std_w = float(np.std(coverages_weighted))
            min_cov_w = float(np.min(coverages_weighted))
            max_cov_w = float(np.max(coverages_weighted))
            median_w = float(np.median(coverages_weighted))
        else:
            avg_w = std_w = min_cov_w = max_cov_w = median_w = 0.0

        # Statistiche unweighted
        if coverages_unweighted:
            avg_uw = float(np.mean(coverages_unweighted))
            std_uw = float(np.std(coverages_unweighted))
            min_cov_uw = float(np.min(coverages_unweighted))
            max_cov_uw = float(np.max(coverages_unweighted))
            median_uw = float(np.median(coverages_unweighted))
        else:
            avg_uw = std_uw = min_cov_uw = max_cov_uw = median_uw = 0.0

        # Calcola numero di individui per questo archivio (tutte le run di quell'archivio)
        archive_pattern = f"runs/{archive_name}/run_*/"
        individuals, avg_inds, std_inds, elapsed_avg_time = load_archived_individuals(archive_pattern, archive_name)
        num_individuals = len(individuals)
        label_coverage = calculate_label_coverage(individuals)

        all_results[archive_name] = {
            "num_runs": len(both_coverages),
            "num_individuals": num_individuals,
            "avg_inds": avg_inds,
            "std_inds": std_inds,
            "avg -/+ std": (avg_inds - std_inds, avg_inds + std_inds),
            "coverages_weighted": coverages_weighted,
            "coverages_unweighted": coverages_unweighted,
            "weighted": {"average": avg_w, "std": std_w, "min": min_cov_w, "max": max_cov_w, "median": median_w},
            "unweighted": {"average": avg_uw, "std": std_uw, "min": min_cov_uw, "max": max_cov_uw, "median": median_uw},
            "label_coverage": label_coverage,
            "avg_run_time": elapsed_avg_time,
        }

        # Aggiungi alla lista per la tabella
        results_list.append(
            {
                "Archive": archive_name,
                "Runs": len(both_coverages),
                "Total Individuals": num_individuals,
                "Avg ± Std Inds per Run": f"{round(float(avg_inds), 2)} ± {round(float(std_inds), 2)}",
                "Avg ± Std Weighted Coverage (%)": f"{round(avg_w, 2)} ± {round(std_w, 2)}",
                "Avg ± Std Unweighted Coverage(%)": f"{round(avg_uw, 2)} ± {round(std_uw, 2)}",
                "Label Coverage (%)": round(label_coverage, 2),
                "Avg Run Time(s)": elapsed_avg_time,
            }
        )

    # Salva risultati in json
    with open(f"{DIVERSITY_OUTPUT_FOLDER}/coverages.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Crea e salva tabella
    df = pd.DataFrame(results_list)
    df.to_csv(f"{DIVERSITY_OUTPUT_FOLDER}/coverages_summary.csv", index=False)

    # Stampa tabella
    print(f"COMPARATIVE SUMMARY")
    print(f"(Each run vs all other runs from all archives)")
    print(df.to_string(index=False))

    print(f"Coverage results saved to {DIVERSITY_OUTPUT_FOLDER}/coverages.json")
    print(f"Summary table saved to {DIVERSITY_OUTPUT_FOLDER}/coverages_summary.csv")


def archive_main():
    inds_data = []  # Store individuals data
    method_names = []  # Store corresponding method used
    num_individuals_per_method = {}  # Count individuals per method
    label_coverages = {}  # Store label coverage per method

    # Load data from all methods
    for method_name, folder_pattern in METHODS_TO_COMPARE.items():
        individuals, _, _, _ = load_archived_individuals(folder_pattern, method_name)
        inds_data.extend(individuals)
        num_individuals_per_method[method_name] = len(individuals)
        if len(individuals) > 0:
            method_names.append(method_name)

        coverage = calculate_label_coverage(individuals)
        label_coverages[method_name] = coverage

    if len(inds_data) == 0:
        print("ERROR: No data loaded!")
        exit(1)

    # Calculate distance matrix once for all individuals
    distance_matrix = compute_individual_distance_matrix(inds_data)

    # Analisi
    coverage_dict_w, coverage_dict_uw, num_clusters = compute_coverage_and_plot(
        inds_data, method_names, distance_matrix
    )

    # Prepara dati per salvataggio
    results_data = {
        "num_clusters": num_clusters,
        "total_samples": len(inds_data),
        "coverage_weighted": coverage_dict_w,
        "coverage_unweighted": coverage_dict_uw,
        "label_coverages": label_coverages,
    }

    # Aggiungi numero di individui per ogni archivio
    results_list = []
    for method_name, folder_pattern in METHODS_TO_COMPARE.items():
        num_individuals = num_individuals_per_method.get(method_name, 0)
        coverage_w = coverage_dict_w.get(method_name, 0.0)
        coverage_uw = coverage_dict_uw.get(method_name, 0.0)
        label_coverage = label_coverages.get(method_name, 0.0)

        results_data[f"{method_name}_individuals"] = num_individuals

        results_list.append(
            {
                "Archive": method_name,
                "Total Individuals": num_individuals,
                "Coverage Weighted (%)": round(coverage_w, 2),
                "Coverage Unweighted (%)": round(coverage_uw, 2),
                "Label Coverage (%)": round(label_coverage, 2),
            }
        )

    # Salva risultati in JSON
    with open(f"{DIVERSITY_OUTPUT_FOLDER}/archive_coverages.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Salva tabella in CSV
    df = pd.DataFrame(results_list)
    df.to_csv(f"{DIVERSITY_OUTPUT_FOLDER}/archive_coverages_summary.csv", index=False)

    # Stampa tabella
    print("\n=== FINAL RESULTS ===")
    print(f"Total clusters: {num_clusters}")
    print(f"Total samples: {len(inds_data)}")
    print("Coverage by Archive:")
    print(df.to_string(index=False))

    print(f"Results saved to {DIVERSITY_OUTPUT_FOLDER}/")
    print(f"  - archive_coverages.json")
    print(f"  - archive_coverages_summary.csv")


if __name__ == "__main__":
    if ANALYSIS_CONFIG == "single_run":
        single_run_main()
    elif ANALYSIS_CONFIG == "archives":
        archive_main()
    else:
        raise ValueError(f"Unknown ANALYSIS_CONFIG: {ANALYSIS_CONFIG}")
