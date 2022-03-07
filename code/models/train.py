import argparse
from unicodedata import name

from azureml.core import Run, Dataset
import mlflow
import mlflow.tensorflow

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def parse_args() -> argparse.Namespace:
    """Parses script defined input arguments

    Returns:
        argparse.Namespace: Namespace with values of all parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=str, dest="dataset_dir", help="mounted dataset directory")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, dest="epochs", default=10, help="number of training epochs")

    return parser.parse_args()


def main(args: argparse.Namespace):
    # get Azure context and WorkSpace
    run = Run.get_context()
    ws = run.experiment.workspace

    # # define input image size
    img_height = 224
    img_width = 224
    img_layers = 3

    # # prepare datasets
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    # Datasets performance tuning
    num_classes = len(train_ds.class_names)
    print(f'Number of classes: {num_classes:n}')
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # MLFlow autologger
    mlflow.tensorflow.autolog()

    # define TF model
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, img_layers)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # train process
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)
