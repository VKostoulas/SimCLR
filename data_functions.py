import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import elasticdeform.tf as etf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def prepare_dataset(dataset_name, unlabeled_dataset_split, labeled_dataset_split, test_dataset_split, batch_size):
    # Labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly

    unlabeled_train_dataset, unlabeled_metadata = tfds.load(dataset_name, split=unlabeled_dataset_split,
                                                            as_supervised=True, shuffle_files=True, with_info=True)
    num_of_unlabeled_examples = unlabeled_metadata.splits[unlabeled_dataset_split].num_examples
    print(f"Unlabeled train dataset from '{dataset_name}' has {num_of_unlabeled_examples} examples")

    labeled_train_dataset, labeled_metadata = tfds.load(dataset_name, split=labeled_dataset_split, as_supervised=True,
                                                        shuffle_files=True, with_info=True)
    num_of_labeled_examples = labeled_metadata.splits[labeled_dataset_split].num_examples
    print(f"Labeled train dataset from '{dataset_name}' has {num_of_labeled_examples} examples")

    test_dataset, test_metadata = tfds.load(dataset_name, split=test_dataset_split, as_supervised=True, with_info=True)
    num_of_test_examples = test_metadata.splits[test_dataset_split].num_examples
    print(f"Validation dataset from '{dataset_name}' has {num_of_test_examples} examples")

    steps_per_epoch = (num_of_unlabeled_examples + num_of_labeled_examples) // batch_size
    unlabeled_batch_size = num_of_unlabeled_examples // steps_per_epoch
    labeled_batch_size = num_of_labeled_examples // steps_per_epoch
    print(f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)")

    unlabeled_train_dataset = unlabeled_train_dataset\
        .shuffle(buffer_size=10 * unlabeled_batch_size)\
        .batch(unlabeled_batch_size)

    labeled_train_dataset = labeled_train_dataset\
        .shuffle(buffer_size=10 * labeled_batch_size)\
        .batch(labeled_batch_size)

    test_dataset = test_dataset\
        .batch(batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, labeled_train_dataset, test_dataset


# Distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness)
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter)

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices)
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


class RandomColorDistortion(layers.Layer):
    def __init__(self, brightness=0, contrast_low=0, contrast_up=0, saturation_low=0, saturation_up=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.contrast_low = contrast_low
        self.contrast_up = contrast_up
        self.saturation_low = saturation_low
        self.saturation_up = saturation_up

    def call(self, images, training=True):
        if training:
            images = tf.image.random_brightness(images, self.brightness)
            images = tf.image.random_saturation(images, self.saturation_low, self.saturation_up)
            images = tf.image.random_saturation(images, self.contrast_low, self.contrast_up)
            images = tf.clip_by_value(images, 0, 1)
        return images


class GaussianBlur(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, images, training=True):
        if training:
            if tf.random.uniform(shape=[1, 1])[0][0] <= 0.5:  # 50% chance to apply gaussian blur
                images = tfa.image.gaussian_filter2d(images, sigma=0.5)
        return images


class ElasticDeform(layers.Layer):
    def __init__(self, prob=1.0, **kwargs):
        super().__init__(**kwargs)

        self.prob = prob

    def call(self, images, training=True):
        if training:
            if tf.random.uniform(shape=[1, 1])[0][0] <= self.prob:  # probability to apply elastic deformation
                img_shape = images.shape[1:]
                displacement = tf.random.normal((2, 10, 10))
                images = etf.deform_grid(images, displacement, mode='nearest', axis=(1, 2))
                images = tf.clip_by_value(images, 0., 1.)
                images = tf.reshape(images, [-1, *img_shape])  # just to know the shape of output, else produces error
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, contrast_low, contrast_up, saturation_low, saturation_up, elastic_prob,
                  image_size, image_channels, resize_shape=None):
    zoom_factor = 1.0 - tf.sqrt(min_area)
    block = tf.keras.Sequential([tf.keras.Input(shape=(image_size, image_size, image_channels))])
    if resize_shape:
        block.add(preprocessing.Resizing(*resize_shape))
    block.add(preprocessing.Rescaling(1 / 255))
    block.add(ElasticDeform(elastic_prob))
    block.add(preprocessing.RandomFlip("horizontal_and_vertical"))  # this only for medical data, else "horizontal"
    # block.add(preprocessing.RandomTranslation(zoom_factor / 2, zoom_factor / 2))
    # block.add(preprocessing.RandomRotation(zoom_factor/2))
    block.add(preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)))
    # block.add(RandomColorAffine(brightness, jitter))
    block.add(RandomColorDistortion(brightness, contrast_low, contrast_up, saturation_low, saturation_up))
    # block.add(GaussianBlur())

    return block


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

