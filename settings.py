# Dataset parameters
dataset_name = "patch_camelyon"
unlabeled_dataset_split = "train[:40%]"
labeled_dataset_split = "train[40%:50%]"
test_dataset_split = "validation[:50%]"

# Dataset hyperparameters
image_size = 96
resize_size = None
image_channels = 3
num_of_classes = 2

# Algorithm hyperparameters
num_epochs = 20
batch_size = 400
width = 1024
temperature = 0.1

# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.25,  "brightness": 0.2, "contrast_low": 0.8, "contrast_up": 2.0,
                            "saturation_low": 0.7, "saturation_up": 1.4, "elastic_prob": 0.5,
                            "image_size": image_size, "image_channels": image_channels, "resize_shape": resize_size}
classification_augmentation = {"min_area": 0.55, "brightness": 0.2, "contrast_low": 0.8, "contrast_up": 2.0,
                               "saturation_low": 0.7, "saturation_up": 1.4, "elastic_prob": 0.5,
                               "image_size": image_size, "image_channels": image_channels, "resize_shape": resize_size}
