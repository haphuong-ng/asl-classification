# Import lib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input, layers, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore, CSVLogger)
from sklearn.metrics import precision_score, recall_score, f1_score

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.20
SPLIT_SEED = 42
EPOCHS = 100

# Load data
DATASET_DIR = Path("/kaggle/input/datasets/lexset/synthetic-asl-alphabet")
TRAIN_DIR = DATASET_DIR / "Train_Alphabet"
TEST_DIR  = DATASET_DIR / "Test_Alphabet"

# Remove "J", "Z" classes
class_names = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir() and p.name not in {"J", "Z"})
NUM_CLASSES = len(class_names)

# Train/Val/Test Split
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VAL_RATIO,
    subset="training",
    seed=SPLIT_SEED,
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    class_names=class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VAL_RATIO,
    subset="validation",
    seed=SPLIT_SEED,
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    class_names=class_names)

# Giả định Test_Alphabet cũng chứa các thư mục con theo class.
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    class_names=class_names
)
AUTOTUNE = tf.data.AUTOTUNE

# Data Pipeline
# scaling
rescale = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y),num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (rescale(x), y))
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

#Autotune & prefetch
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(factor=0.05),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomZoom(height_factor=0.1),
    layers.RandomContrast(factor=0.2),
    layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),], name="data_augmentation")

# Model
L2_REG = 1e-4

# CONVOLUTION BLOCK
def conv_bn_relu(x, filters, kernel_size):
    x = layers.Conv2D(filters=filters,kernel_size=kernel_size,padding="same",use_bias=False,kernel_regularizer=regularizers.l2(L2_REG),)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

# INCEPTION BLOCK
def inception_block(x, out_filters):
    f = out_filters // 4

    # Branch 1: 1x1 convolution
    b1 = conv_bn_relu(x, f, 1)

    # Branch 2: 1x1 -> 3x3
    b2 = conv_bn_relu(x, f, 1)
    b2 = conv_bn_relu(b2, f, 3)

    # Branch 3: 1x1 -> 5x5
    b3 = conv_bn_relu(x, f, 1)
    b3 = conv_bn_relu(b3, f, 5)

    # Branch 4: MaxPooling -> 1x1
    b4 = layers.MaxPool2D(pool_size=3,strides=1,padding="same",)(x)
    b4 = conv_bn_relu(b4, f, 1)

    # Concatenate along channel axis
    return layers.Concatenate(axis=-1)([b1, b2, b3, b4])

# BUILD INCEPTION CNN
def multi_scale_inception(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    # Input
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # Data Augmentation
    x = data_augmentation(inputs)

    # Initial convolution
    x = conv_bn_relu(x, 16, 3)
    x = layers.MaxPool2D(pool_size=2,name="initial_maxpool")(x)

    # Inception Block 1
    x = inception_block(x, 64)
    x = layers.MaxPool2D(pool_size=2,name="intermediate_maxpool")(x)

    # Inception Block 2
    x = inception_block(x, 224)

    # Final convolution
    x = conv_bn_relu(x, 256, 3)
    x = layers.SpatialDropout2D(0.20,name="spatial_dropout")(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Fully connected classifier
    x = layers.Dropout(0.40,name="classifier_dropout")(x)

    # Output layer
    outputs = layers.Dense(num_classes,activation="softmax",name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="multi_scale_inception")

model = multi_scale_inception(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callback
MODEL_NAME = "multi_scale_inception"
RUN_ID = 3
OUTPUT_DIR = Path("/kaggle/working") / MODEL_NAME / f"run_{RUN_ID}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(OUTPUT_DIR / "best_model.keras")

callbacks = [
    BackupAndRestore(backup_dir=str(OUTPUT_DIR / "backup")),
    CSVLogger(filename=str(OUTPUT_DIR / "history.csv")),
    ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_delta=0.001, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_accuracy", mode="max", min_delta=0.001, patience=6, restore_best_weights=True, verbose=1),
]

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# Evaluate
best_model = tf.keras.models.load_model(MODEL_PATH)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=2)

y_true, y_pred = [], []
for images, labels in test_ds:
    predictions = best_model(images, training=False)
    y_true.extend(labels.numpy().reshape(-1))
    y_pred.extend(np.argmax(predictions.numpy(), axis=1))
    
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

test_metrics = pd.DataFrame([{
    "model": MODEL_NAME,
    "run": RUN_ID,
    "accuracy": test_acc,
    "precision_macro": precision,
    "recall_macro": recall,
    "f1_macro": f1,
    "loss": test_loss
}])

test_metrics.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

# Training curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title(f"{MODEL_NAME} - Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title(f"{MODEL_NAME} - Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "training_curve.png")
plt.close(fig)