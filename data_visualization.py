from PIL import Image  # Actually Needed to export_as_gif
import matplotlib.pyplot as plt


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

    if rubber_band:
        pil_images += pil_images[2:-1][::-1]

    pil_images[0].save(
        filename,
        save_all=True,
        append_images=pil_images[1:],
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
    ax.set_xlabel("Mutations")
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
