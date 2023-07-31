from matplotlib import pyplot as plt


def plot_images(images_arr):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(8, 8))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def plot_images_with_labels(images_arr, predicted_labels, labels, classes: dict):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(8, 8))
    axes = axes.flatten()
    for img, ax, label, p_label in zip(images_arr, axes, labels, predicted_labels):
        if label == p_label:
            color = 'blue'
        else:
            color = 'red'
        ax.set_title(classes[p_label], color=color)
        ax.imshow(img)
    plt.tight_layout()
    plt.show()