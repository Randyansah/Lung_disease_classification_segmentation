import pathlib
import tensorflow as tf
from PIL import Image
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras


print(tf.version.VERSION)

data_dir=None

train_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(None,None),
    batch_size=32
)
dataset_classnames=train_ds.class_names
print(f"Dataset class names are:{dataset_classnames}")

val_ds = tf.keras.utils.image_dataset_from_directory(

    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(dataset_classnames)


