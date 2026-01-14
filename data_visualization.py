from PIL import Image  # Actually Needed to export_as_gif
import matplotlib.pyplot as plt


def export_as_gif(filename, images, frames_per_second=5, rubber_band=False):
    """
    Create a gif from all the images generated in the process

    Args:
        filename (_type_): _description_
        images (_type_): _description_
        frames_per_second (int, optional): _description_. Defaults to 5.
        rubber_band (bool, optional): _description_. Defaults to False.
    """
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


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
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Confidence Score")
    ax.tick_params("y")

    # Set title
    plt.title("Confidence Scores")

    # Save plot
    plt.savefig(save_path)
    plt.close(fig)

    # Print message
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

    # Set title
    plt.title(title)

    # Save plot
    plt.savefig(save_path)
    plt.close(fig)

    # Print message
    print(f"Plot saved to {save_path}")
