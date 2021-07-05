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
num_epochs = 70
batch_size = 400
width = 1024
temperature = 0.1

# Stronger augmentations for contrastive, weaker ones for supervised training
# contrastive_augmentation = {"min_area": 0.1,  "brightness": 0.8, "contrast": [0.2, 2.0], "saturation": [0.2, 1.8],
#                             "hue": 0.2, "elastic_prob": 0., "color_prob": 0.8, "image_size": image_size,
#                             "image_channels": image_channels, "resize_shape": resize_size}
# classification_augmentation = {"min_area": 0.81, "brightness": 0.3, "contrast": [0.8, 1.8], "saturation": [0.7, 1.4],
#                                "hue": 0.1, "elastic_prob": 0., "color_prob": 0.8, "image_size": image_size,
#                                "image_channels": image_channels, "resize_shape": resize_size}

contrastive_augmentation = {"min_area": 0.1,  "brightness": 0.3, "contrast": [0.5, 2.0], "saturation": [0.5, 1.8],
                            "hue": 0.2, "elastic_prob": 0., "color_prob": 1.0}
classification_augmentation = {"min_area": 0.81, "brightness": 0.2, "contrast": [0.8, 1.8], "saturation": [0.7, 1.4],
                               "hue": 0.1, "elastic_prob": 0., "color_prob": 0.8}
