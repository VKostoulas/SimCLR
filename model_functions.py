import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def inception_encode_step(input_volume, filters, conv_args, reduce_dims, name):
    """

    :param input_volume:
    :param filters:
    :param conv_args:
    :param reduce_dims:
    :param name:
    :return:
    """
    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_prod = filters
    block1 = keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_1x1')
    block2 = keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_3x3')
    block3 = keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_5x5')
    block4 = keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=name + '_path_pool')

    block1.add(layers.Conv2D(filters_1x1, kernel_size=(1, 1), strides=2, **conv_args))
    block1.add(layers.BatchNormalization())
    block1.add(layers.PReLU())
    block1.add(layers.SpatialDropout2D(0.2))
    block4.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))

    if reduce_dims:
        block2.add(layers.Conv2D(filters_3x3_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block2.add(layers.BatchNormalization())
        block2.add(layers.PReLU())
        block2.add(layers.SpatialDropout2D(0.2))
        block3.add(layers.Conv2D(filters_5x5_reduce, kernel_size=(1, 1), strides=1, **conv_args))
        block3.add(layers.BatchNormalization())
        block3.add(layers.PReLU())
        block3.add(layers.SpatialDropout2D(0.2))
        block4.add(layers.Conv2D(filters_pool_prod, kernel_size=(1, 1), strides=1, **conv_args))
        block4.add(layers.BatchNormalization())
        block4.add(layers.PReLU())
        block4.add(layers.SpatialDropout2D(0.2))

    block2.add(layers.Conv2D(filters_3x3, kernel_size=(3, 3), strides=2, **conv_args))
    block2.add(layers.BatchNormalization())
    block2.add(layers.PReLU())
    block2.add(layers.SpatialDropout2D(0.2))
    block3.add(layers.Conv2D(filters_5x5, kernel_size=(5, 5), strides=2, **conv_args))
    block3.add(layers.BatchNormalization())
    block3.add(layers.PReLU())
    block3.add(layers.SpatialDropout2D(0.2))

    layer_outputs = [block1(input_volume), block2(input_volume), block3(input_volume), block4(input_volume)]
    concat = layers.Concatenate(axis=-1, name=name + '_output')(layer_outputs)

    return concat


def inception_network(image_size, image_channels, width, kernel_initializer='glorot_uniform'):

    inputs = layers.Input(shape=(image_size, image_size, image_channels))
    conv_args = {'kernel_initializer': kernel_initializer, 'use_bias': True, 'padding': 'same'}

    x = inception_encode_step(inputs, [16, 96, 64, 16, 32, 16], conv_args, reduce_dims=True, name='inc_enc_block_1')
    x = inception_encode_step(x, [32, 96, 128, 16, 64, 32], conv_args, reduce_dims=True, name='inc_enc_block_2')
    x = inception_encode_step(x, [64, 96, 256, 16, 128, 32], conv_args, reduce_dims=True, name='inc_enc_block_3')
    x = inception_encode_step(x, [128, 128, 512, 32, 128, 64], conv_args, reduce_dims=True, name='inc_enc_block_4')

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(width)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='inception_auto_encoder')


def residual_encode_step(input_volume, filters, kernel_size, apply_batch_norm, names, conv_args):
    """
    Down step for advanced_auto_encoder.

    :param input_volume:
    :param filters:
    :param kernel_size:
    :param apply_batch_norm:
    :param names:
    :param conv_args
    :return:
    """
    # TODO: Use bias should be the same for all here ?
    use_bias = not apply_batch_norm

    # Main path
    encode_block = tf.keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=names[0])
    if apply_batch_norm:
        encode_block.add(layers.BatchNormalization())
    encode_block.add(layers.LeakyReLU(0.2))
    encode_block.add(layers.AveragePooling2D(2))
    encode_block.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))
    if apply_batch_norm:
        encode_block.add(layers.BatchNormalization())
    encode_block.add(layers.LeakyReLU(0.2))
    encode_block.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias, **conv_args))

    # Residual path
    residual_block = tf.keras.Sequential([layers.InputLayer(input_shape=input_volume.shape[1:])], name=names[1])
    residual_block.add(layers.AveragePooling2D(2))
    # if the main path and residual path have not the same shapes, we add an 1x1 convolution
    if encode_block(input_volume).shape[1:] != residual_block(input_volume).shape[1:]:
        residual_block.add(layers.Conv2D(filters=filters, kernel_size=1, use_bias=True, **conv_args))

    # Build paths and add them
    encode_path = encode_block(input_volume)
    residual_path = residual_block(input_volume)
    added_outputs = layers.Add()([encode_path, residual_path])

    return added_outputs


# Define the encoder architecture
def get_encoder(image_size, image_channels, width):
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(width),
            layers.BatchNormalization(),
            layers.PReLU(),
        ],
        name="encoder",
    )


def get_projection_head(width):
    return keras.Sequential(
            [
                keras.Input(shape=(width,)),
                layers.Dense(width/2),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Dense(width/4),
            ],
            name="projection_head",
           )


# Define the contrastive model with model-subclassing
class ContrastiveModel(keras.Model):
    def __init__(self, temperature, augment_function, contrastive_augmentation, classification_augmentation,
                 image_size, image_channels, width, num_of_classes):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = augment_function(**contrastive_augmentation)
        self.classification_augmenter = augment_function(**classification_augmentation)
        self.encoder = get_encoder(image_size=image_size, image_channels=image_channels, width=width)
        # Non-linear MLP as projection head
        self.projection_head = get_projection_head(width)

        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(num_of_classes)], name="linear_probe"
        )

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Labels are only used in evaluation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(labeled_images)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
