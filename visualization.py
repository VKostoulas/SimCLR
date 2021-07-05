import matplotlib.pyplot as plt
from data_functions import get_augmenter, augment_image


def visualize_augmentations(num_images, train_dataset, classification_augmentation, contrastive_augmentation):
    # Sample a batch from a dataset
    images = next(iter(train_dataset))[0][0][:num_images]
    # Apply augmentations
    augmented_images = zip(
        images,
        get_augmenter(**classification_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
        get_augmenter(**contrastive_augmentation)(images),
    )
    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
        "Strongly augmented:",
    ]
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(image)
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_tf_augmentations(num_images, test_dataset, classification_augmentation, contrastive_augmentation):
    # Sample a batch from a dataset
    curr_batch = next(iter(test_dataset))
    images, labels = curr_batch[0][:num_images], curr_batch[1][:num_images]
    # Apply augmentations
    soft_augmentations = [augment_image((image, label), **classification_augmentation)[0] for (image, label)
                          in zip(images, labels)]
    strong_augmentations1 = [augment_image((image, label), **contrastive_augmentation)[0] for (image, label)
                             in zip(images, labels)]
    strong_augmentations2 = [augment_image((image, label), **contrastive_augmentation)[0] for (image, label)
                             in zip(images, labels)]
    augmented_images = zip(images, soft_augmentations, strong_augmentations1, strong_augmentations2)

    row_titles = [
        "Original:",
        "Weakly augmented:",
        "Strongly augmented:",
        "Strongly augmented:",
    ]
    plt.figure(figsize=(num_images * 2.2, 4 * 2.2), dpi=100)
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, num_images, row * num_images + column + 1)
            plt.imshow(image)
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


# The classification accuracies of the baseline and the pretraining + finetuning process:
def plot_training_curves(pretraining_history, finetuning_history, baseline_history):
    for metric_key, metric_name in zip(["acc", "loss"], ["accuracy", "loss"]):
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(
            baseline_history.history[f"val_{metric_key}"], label="supervised baseline"
        )
        plt.plot(
            pretraining_history.history[f"val_p_{metric_key}"],
            label="self-supervised pretraining",
        )
        plt.plot(
            finetuning_history.history[f"val_{metric_key}"],
            label="supervised finetuning",
        )
        plt.legend()
        plt.title(f"Classification {metric_name} during training")
        plt.xlabel("epochs")
        plt.ylabel(f"validation {metric_name}")
        plt.show()
        plt.close()