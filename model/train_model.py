import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BASE_DIR = r"C:\PlantAI\model\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def get_label(class_name):
    if "healthy" in class_name.lower():
        return "healthy"
    return "diseased"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

all_classes = sorted(os.listdir(TRAIN_DIR))
class_mapping = {cls: get_label(cls) for cls in all_classes}

with open("class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=all_classes
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=all_classes
)

train_labels = np.array([0 if class_mapping[cls] == "healthy" else 1 
                          for cls, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1])])

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint("plant_model.h5", save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
]

def custom_generator(generator, class_map):
    for batch_imgs, batch_labels in generator:
        new_labels = np.array([
            0 if class_map[list(generator.class_indices.keys())[
                list(generator.class_indices.values()).index(int(l))]] == "healthy" else 1
            for l in batch_labels
        ])
        yield batch_imgs, new_labels

train_steps = train_generator.samples // BATCH_SIZE
valid_steps = valid_generator.samples // BATCH_SIZE

model.fit(
    custom_generator(train_generator, class_mapping),
    steps_per_epoch=train_steps,
    validation_data=custom_generator(valid_generator, class_mapping),
    validation_steps=valid_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Training complete. Model saved as plant_model.h5")