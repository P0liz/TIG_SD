# %%writefile data_visualization.py
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from venn import venn

from config import PROMPTS, ANALYSIS_CONFIG, DIVERSITY_OUTPUT_FOLDER, FOCUS_NAME, OTHERS_NAME


def export_as_gif(filename, images, frames_per_second=5, rubber_band=False):
    """
    Create a gif from all the images generated in the process

    Args:
        filename: path to save the gif
        images: list of torch.Tensor [1, 1, 28, 28] grayscale
        frames_per_second (int, optional): Defaults to 5.
        rubber_band (bool, optional): Defaults to False.
    """
    pil_images = []
    for img in images:
        # Converting tensor to PIL Image (in grayscale)
        img_np = img.detach().cpu().numpy().squeeze()
        img_np = (img_np * 255).astype("uint8")
        pil_images.append(Image.fromarray(img_np, mode="L"))

    if rubber_band and len(pil_images) > 2:
        pil_images += pil_images[2:-1][::-1]

    pil_images[0].save(
        filename, save_all=True, append_images=pil_images[1:], duration=1000 // frames_per_second, loop=0
    )


def plot_labels(label_history, save_path):
    """
    Plot label distribution over generations.

    Args:
        label_history (dict): {generation: [labels]} mapping
        save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    generations = sorted(label_history.keys())
    counts = {i: [] for i in range(len(PROMPTS))}

    for gen in generations:
        labels = label_history[gen]
        for digit in range(len(PROMPTS)):
            counts[digit].append(labels.count(digit))

    for digit in range(len(PROMPTS)):
        ax.plot(generations, counts[digit], label=f"Digit {digit}", marker="o")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.title("Label Distribution Over Generations")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_confidence(confidences, save_path):
    """
    Plot confidence scores and save the plot to a given path.

    Args:
        confidences (list): List of confidence scores.
        save_path (str): Path to save the plot.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot confidence scores
    ax.plot(confidences, color="b")
    ax.set_xlabel("Mutations")
    ax.set_ylabel("Confidence Score")
    ax.tick_params("y")

    # Save plot
    plt.title("Confidence Scores")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_distance(distances, save_path, title):
    """
    Plot distances.

    Args:
        distances (list): List of distances.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot distances
    ax.plot(distances, color="r")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.tick_params("y")

    # Save plot
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# Chiama la nuova funzione per il plot con expected labels
def plot_diversity_with_expected_labels(
    all_data, cluster_labels, centers_tsne, tsne_results, method_names, coverage_dict, num_clusters, idx=0
):
    """
    Plot t-SNE con colori cluster e edge color per expected label
    """
    labels_method = [sample[1] for sample in all_data]
    expected_labels = [sample[2] for sample in all_data]

    # Mappa cluster a colori
    cluster_colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    # Mappa expected labels a colori edge
    unique_expected = sorted(set(expected_labels))
    expected_colors = {label: plt.cm.Set3(i / len(unique_expected)) for i, label in enumerate(unique_expected)}

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

    # Plot
    plt.figure(figsize=(14, 10))

    # Plot centroids
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

    # Plot per cluster, metodo e expected label
    plotted_methods = set()
    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_color = cluster_colors[cluster_id]

        for method in method_names:
            method_and_cluster_mask = [cluster_mask[i] and labels_method[i] == method for i in range(len(all_data))]

            if any(method_and_cluster_mask):
                label = f"{method} ({coverage_dict[method]:.1f}%)" if method not in plotted_methods else None
                if label:
                    plotted_methods.add(method)

                size = 200 if ANALYSIS_CONFIG == "single_run" and method == "focus" else 80

                # Plot per ogni expected label
                for exp_label in unique_expected:
                    mask_with_exp = [
                        method_and_cluster_mask[i] and expected_labels[i] == exp_label for i in range(len(all_data))
                    ]

                    if any(mask_with_exp):
                        plt.scatter(
                            tsne_results[mask_with_exp, 0],
                            tsne_results[mask_with_exp, 1],
                            c=[cluster_color],
                            marker=method_markers[method],
                            s=size,
                            edgecolors=expected_colors[exp_label],
                            linewidths=2,
                            zorder=2,
                        )

    # Legenda per expected labels
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="w",
            marker="o",
            linestyle="",
            markersize=8,
            markeredgecolor=expected_colors[label],
            markerfacecolor="white",
            markeredgewidth=2,
            label=f"Expected: {label}",
        )
        for label in unique_expected
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

    if ANALYSIS_CONFIG == "single_run":
        plt.title(f"Single Run Diversity with Expected Labels (t-SNE)\nTotal clusters: {num_clusters}")
    else:
        plt.title(f"Archive Diversity with Expected Labels (t-SNE)\nTotal clusters: {num_clusters}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    plt.tight_layout()
    plt.savefig(f"{DIVERSITY_OUTPUT_FOLDER}/diversity_expected_labels_{idx}.pdf")
    plt.show()


def plot_coverage_venn(inds_data, cluster_labels, method_names):
    """
    Crea un diagramma di Venn per mostrare overlap di coverage tra 4 archivi.
    """
    assert ANALYSIS_CONFIG == "archives", "Venn diagram is only for archives comparison"

    if len(method_names) != 4:
        print(f"Venn diagram requires exactly 4 methods, got {len(method_names)}")
        return None

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    labels_method = [sample[0] for sample in inds_data]

    # Calcola quali cluster copre ogni metodo
    method_clusters = {}
    for method in method_names:
        method_indices = [i for i, label in enumerate(labels_method) if label == method]
        method_clusters[method] = set(cluster_labels[i] for i in method_indices if cluster_labels[i] != -1)

    set_A = method_clusters[method_names[0]]
    set_B = method_clusters[method_names[1]]
    set_C = method_clusters[method_names[2]]
    set_D = method_clusters[method_names[3]]

    sets_dict = {method_names[0]: set_A, method_names[1]: set_B, method_names[2]: set_C, method_names[3]: set_D}

    plt.figure(figsize=(14, 10))

    venn(sets_dict, fmt="{size}\n({percentage:.1f}%)", fontsize=8)

    plt.title(
        f"Archive Coverage Overlap (4 Methods)\nTotal Clusters: {num_clusters}", fontsize=16, fontweight="bold", pad=20
    )

    # Calcolo statistiche
    union_all = set_A | set_B | set_C | set_D
    intersection_all = set_A & set_B & set_C & set_D

    stats_text = (
        f"COVERAGE STATISTICS\n"
        f"{'='*40}\n\n"
        f"Individual Coverage:\n"
        f"  • {method_names[0]}: {len(set_A)} ({len(set_A)/num_clusters*100:.1f}%)\n"
        f"  • {method_names[1]}: {len(set_B)} ({len(set_B)/num_clusters*100:.1f}%)\n"
        f"  • {method_names[2]}: {len(set_C)} ({len(set_C)/num_clusters*100:.1f}%)\n"
        f"  • {method_names[3]}: {len(set_D)} ({len(set_D)/num_clusters*100:.1f}%)\n\n"
        f"Union (coperti da almeno uno): {len(union_all)} "
        f"({len(union_all)/num_clusters*100:.1f}%)\n"
        f"Shared by all archives: {len(intersection_all)} "
        f"({len(intersection_all)/num_clusters*100:.1f}%)\n"
    )

    plt.text(
        1.05,
        0.5,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, pad=1),
    )

    plt.tight_layout()
    plt.savefig(f"{DIVERSITY_OUTPUT_FOLDER}/coverage_venn.png", bbox_inches="tight", dpi=300)
    plt.close()
