import argparse

from azureml.core import Run, Dataset
import mlflow

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# define input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, dest='dataset_name', help='dataset used to train name')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--epochs', type=float, dest='epochs', default=10, help='number of training epochs')

args = parser.parse_args()

# get Azure context and WorkSpace
run = Run.get_context()
ws = run.experiment.workspace

# get input dataset by name and mount it
dataset = Dataset.get_by_name(ws, name=args.dataset_name)
# dataset_input = DatasetConsumptionConfig("input_1", file_pipeline_param).as_mount()
data_dir = 'dataset'
dataset.download(target_path=data_dir, overwrite=False)

# dataset = dataset.as_mount()
# mount_point = run.input_datasets['input_1']


# # define input image size
img_height = 224
img_width = 224
img_layers = 3

# # prepare datasets
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Datasets performance tuning
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# MLFlow autologger
mlflow.tensorflow.autolog()

# define TF model
num_classes = len(train_ds.class_names)

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.Accuracy()])


# train process
with mlflow.start_run(run_name='simple CNN'):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )
