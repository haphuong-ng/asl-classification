from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
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

class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

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

# augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(factor=0.05),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomZoom(height_factor=0.1),
    layers.RandomContrast(factor=0.2),
    layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),], name="data_augmentation")

# Model
def build_simple_cnn(img_size=224, num_classes=10):
    inputs = Input(shape=(img_size, img_size, 3), name="input_image")
    
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_classification")(x)

    return Model(inputs=inputs, outputs=outputs, name="simple_cnn")

model = build_simple_cnn(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callback
MODEL_NAME = "simple_cnn"
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