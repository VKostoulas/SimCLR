import os

from python_settings import settings as s
import settings as local_settings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_functions import prepare_dataset, prepare_tf_dataset, zip_and_prefetch_datasets
from visualization import visualize_augmentations, visualize_tf_augmentations, get_augmenter, plot_training_curves
from model_functions import get_encoder, ContrastiveModel


if __name__ == '__main__':

    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    s.configure(local_settings)

    # train_dataset, labeled_train_dataset, test_dataset = prepare_dataset(dataset_name=s.dataset_name,
    #                                                                      unlabeled_dataset_split=s.unlabeled_dataset_split,
    #                                                                      labeled_dataset_split=s.labeled_dataset_split,
    #                                                                      test_dataset_split=s.test_dataset_split,
    #                                                                      batch_size=s.batch_size)

    dataset_args = {"dataset_name": s.dataset_name, "unlabeled_dataset_split": s.unlabeled_dataset_split,
                    "labeled_dataset_split": s.labeled_dataset_split, "test_dataset_split": s.test_dataset_split,
                    "batch_size": s.batch_size, "classification_augment": s.classification_augmentation,
                    "contrastive_augment": s.contrastive_augmentation}

    train_dataset, labeled_train_dataset, test_dataset = prepare_tf_dataset(**dataset_args)

    visualize_tf_augmentations(num_images=8, test_dataset=test_dataset,
                               classification_augmentation=s.classification_augmentation,
                               contrastive_augmentation=s.contrastive_augmentation)

    # Baseline supervised training with random initialization
    baseline_model = keras.Sequential(
        [
            keras.Input(shape=(s.image_size, s.image_size, s.image_channels)),
            # get_augmenter(**s.classification_augmentation),
            get_encoder(image_size=s.image_size, image_channels=s.image_channels, width=s.width),
            layers.Dense(s.num_of_classes),
        ],
        name="baseline_model",
    )
    baseline_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    baseline_history = baseline_model.fit(labeled_train_dataset, epochs=s.num_epochs, validation_data=test_dataset)
    print("Maximal validation accuracy: {:.2f}%".format(max(baseline_history.history["val_acc"]) * 100))

    # Contrastive pretraining
    pretraining_model = ContrastiveModel(temperature=s.temperature, augment_function=get_augmenter,
                                         contrastive_augmentation=s.contrastive_augmentation,
                                         classification_augmentation=s.classification_augmentation,
                                         image_size=s.image_size, image_channels=s.image_channels, width=s.width,
                                         num_of_classes=s.num_of_classes)
    pretraining_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )

    pretraining_history = pretraining_model.fit(train_dataset, epochs=s.num_epochs, validation_data=test_dataset)
    print("Maximal validation accuracy: {:.2f}%".format(max(pretraining_history.history["val_p_acc"]) * 100))

    # Supervised finetuning of the pretrained encoder
    finetuning_model = keras.Sequential(
        [
            layers.Input(shape=(s.image_size, s.image_size, s.image_channels)),
            get_augmenter(**s.classification_augmentation),
            pretraining_model.encoder,
            layers.Dense(s.num_of_classes),
        ],
        name="finetuning_model",
    )
    finetuning_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    finetuning_history = finetuning_model.fit(labeled_train_dataset, epochs=s.num_epochs, validation_data=test_dataset)
    print("Maximal validation accuracy: {:.2f}%".format(max(finetuning_history.history["val_acc"]) * 100))

    plot_training_curves(pretraining_history, finetuning_history, baseline_history)
