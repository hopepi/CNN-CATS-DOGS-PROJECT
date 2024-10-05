from fileinput import filename
from pathlib import Path
import numpy as np
import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.initializers import HeNormal
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

num_cores = 4
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)


base_dir = Path("C:/Users/umutk/OneDrive/Masaüstü/CatsAndDogs")
train_dir = base_dir / "train"
test_dir = base_dir / "test"
validation_dir = base_dir / "validation"


def create_dataframe(directory):
    filepaths = []
    labels = []

    for label_dir in directory.iterdir():
        if label_dir.is_dir():
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    filepaths.append(str(img_file))
                    labels.append(label_dir.name)

    if not filepaths or not labels:
        raise ValueError("Filepaths or labels are empty.")
    if len(filepaths) != len(labels):
        raise ValueError("Filepaths and labels must have the same length.")

    data = {'Filepath': filepaths, 'Label': labels}
    df = pd.DataFrame(data)
    return df

def create_cnn_model(input_shape,num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax', kernel_initializer=HeNormal()))

    return model

train_df = create_dataframe(train_dir)
test_df = create_dataframe(test_dir)
validation_df = create_dataframe(validation_dir)

input_shape = (180, 180, 3)
num_classes = len(train_df['Label'].unique())
print(train_df['Label'].unique())
print(num_classes)

optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    fill_mode = "nearest"
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    color_mode='rgb',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=32,
    shuffle=True,
    seed = 0,
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=32
)

cnn_model.fit(
    train_generator,# Eğitim verileri
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(180, 180),
    class_mode='sparse',
    batch_size=32,
    color_mode="rgb",
    shuffle=False
)

loss, accuracy = cnn_model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
cnn_model.save('my_model.h5')
"""
%88
"""