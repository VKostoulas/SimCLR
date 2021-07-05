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

    labeled_batch_size, unlabeled_batch_size = define_train_batch_sizes(batch_size, num_of_labeled_examples,
                                                                        num_of_unlabeled_examples)

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


def prepare_tf_dataset(dataset_name, unlabeled_dataset_split, labeled_dataset_split, test_dataset_split,
                       batch_size, classification_augment, contrastive_augment):

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

    labeled_batch_size, unlabeled_batch_size = define_train_batch_sizes(batch_size, num_of_labeled_examples,
                                                                        num_of_unlabeled_examples)

    unlabeled_train_dataset = create_tf_dataset(unlabeled_train_dataset, unlabeled_batch_size, 'train',
                                                contrastive_augment)
    labeled_train_dataset = create_tf_dataset(labeled_train_dataset, labeled_batch_size, 'train', classification_augment)
    test_dataset = create_tf_dataset(test_dataset, batch_size, 'test')

    train_dataset = zip_and_prefetch_datasets(unlabeled_train_dataset, labeled_train_dataset)

    return train_dataset, labeled_train_dataset, test_dataset


def create_tf_dataset(dataset, batch_size, mode, augment_args=None):
    assert mode in ['train', 'test']
    dataset = dataset.shuffle(buffer_size=10 * batch_size) if mode == 'train' else dataset
    dataset = dataset.map(lambda x, y: rescale_image((x, y)), num_parallel_calls=-1)
    if mode == 'train':
        dataset = dataset.map(lambda x, y: augment_image((x, y), **augment_args), num_parallel_calls=-1)
    dataset = dataset.batch(batch_size)
    if mode == 'test':
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def define_train_batch_sizes(batch_size, num_of_labeled_examples, num_of_unlabeled_examples):
    steps_per_epoch = (num_of_unlabeled_examples + num_of_labeled_examples) // batch_size
    unlabeled_batch_size = num_of_unlabeled_examples // steps_per_epoch
    labeled_batch_size = num_of_labeled_examples // steps_per_epoch
    print(f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)")
    return labeled_batch_size, unlabeled_batch_size


def zip_and_prefetch_datasets(unlabeled_train_dataset, labeled_train_dataset):
    # Labeled and unlabeled datasets are zipped together
    train_dataset = unlabeled_train_dataset.concatenate(labeled_train_dataset)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset


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
    def __init__(self, brightness, contrast, saturation, hue, prob, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def call(self, images, training=True):
        if training:
            probability_to_augment = tf.random.uniform(shape=[1, 1])[0][0]
            if probability_to_augment < self.prob or self.prob == 1.0:
                images = tf.image.random_brightness(images, max_delta=self.brightness)
                images = tf.image.random_contrast(images, lower=self.contrast[0], upper=self.contrast[1])
                images = tf.image.random_saturation(images, lower=self.saturation[0], upper=self.saturation[1])
                images = tf.image.random_hue(images, max_delta=self.hue)
                images = tf.clip_by_value(images, 0, 1)
        return images


class GaussianBlur(layers.Layer):
    def __init__(self, prob=0, **kwargs):
        super().__init__(**kwargs)

        self.prob = prob

    def call(self, images, training=True):
        if training:
            probability_to_augment = tf.random.uniform(shape=[1, 1])[0][0]
            if probability_to_augment < self.prob or self.prob == 1.0:
                images = tfa.image.gaussian_filter2d(images, sigma=0.5)
        return images


class ElasticDeform(layers.Layer):
    def __init__(self, prob=0, **kwargs):
        super().__init__(**kwargs)

        self.prob = prob

    def call(self, images, training=True):
        if training:
            probability_to_augment = tf.random.uniform(shape=[1, 1])[0][0]
            if probability_to_augment < self.prob or self.prob == 1.0:
                img_shape = images.shape[1:]
                displacement = tf.random.normal((2, 10, 10))
                images = etf.deform_grid(images, displacement, mode='nearest', axis=(1, 2))
                images = tf.clip_by_value(images, 0., 1.)
                images = tf.reshape(images, [-1, *img_shape])  # just to know the shape of output, else produces error
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, contrast, saturation, hue, elastic_prob, color_prob, image_size, image_channels,
                  resize_shape=None):
    zoom_factor = 1.0 - tf.sqrt(min_area)
    block = tf.keras.Sequential([tf.keras.Input(shape=(image_size, image_size, image_channels))])
    if resize_shape:
        block.add(preprocessing.Resizing(*resize_shape))
    block.add(preprocessing.Rescaling(1 / 255))
    block.add(preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)))
    block.add(preprocessing.RandomFlip("horizontal_and_vertical"))  # this only for medical data, else "horizontal"
    # block.add(preprocessing.RandomTranslation(zoom_factor / 2, zoom_factor / 2))
    # block.add(preprocessing.RandomRotation(zoom_factor/2))
    # block.add(RandomColorAffine(brightness, jitter))
    block.add(ElasticDeform(prob=elastic_prob))
    block.add(RandomColorDistortion(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                                    prob=color_prob))
    # block.add(GaussianBlur())

    return block


def rescale_image(x):
    image, *_ = x  # x is image-label pair
    image = tf.cast(image, tf.float32) / 255.
    return (image,) + x[1:]


def augment_image(x, min_area=0, brightness=0, contrast=0, saturation=0, hue=0, elastic_prob=0,
                  color_prob=0, resize_shape=None):
    # Images must be in [0, 1] range before augmentation

    image, *_ = x  # x is image-label pair

    zoom_factor = 1.0 - tf.sqrt(min_area)
    # image = preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0))(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    probability_for_elastic = tf.random.uniform(shape=[1, 1])[0][0]
    if probability_for_elastic < elastic_prob or elastic_prob == 1.0:
        img_shape = image.shape[1:]
        displacement = tf.random.normal((2, 10, 10))
        image = etf.deform_grid(image, displacement, mode='nearest', axis=(1, 2))
        image = tf.clip_by_value(image, 0., 1.)
        image = tf.reshape(image, [-1, *img_shape])  # just to know the shape of output, else produces error

    probability_for_color = tf.random.uniform(shape=[1, 1])[0][0]
    if probability_for_color < color_prob or color_prob == 1.0:
        image = tf.image.random_brightness(image, max_delta=brightness)
        image = tf.image.random_contrast(image, lower=contrast[0], upper=contrast[1])
        image = tf.image.random_saturation(image, lower=saturation[0], upper=saturation[1])
        image = tf.image.random_hue(image, max_delta=hue)
        image = tf.clip_by_value(image, 0, 1)

    return (image,) + x[1:]




