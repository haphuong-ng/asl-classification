import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32
VAL_RATIO = 0.20
EPOCHS = 100

DATASET_DIR = Path("/kaggle/input/datasets/lexset/synthetic-asl-alphabet")
TRAIN_DIR = DATASET_DIR / "Train_Alphabet"
TEST_DIR = DATASET_DIR / "Test_Alphabet"

# Remove signs J and Z
class_names = sorted([folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir() and folder.name not in ["J", "Z"]])
NUM_CLASSES = len(class_names)
filtered_dir = Path("/kaggle/working/filtered_asl")
filtered_train = filtered_dir / "Train_Alphabet"
filtered_test = filtered_dir / "Test_Alphabet"
filtered_train.mkdir(parents=True, exist_ok=True)
filtered_test.mkdir(parents=True, exist_ok=True)

for cls in class_names:
    train_link = filtered_train / cls
    test_link = filtered_test / cls
    if not train_link.exists():
        os.symlink(TRAIN_DIR / cls, train_link)
    if not test_link.exists():
        os.symlink(TEST_DIR / cls, test_link)

all_ds = tf.keras.utils.image_dataset_from_directory(filtered_train, shuffle=False, image_size=(IMG_SIZE, IMG_SIZE), batch_size=None, label_mode="int", color_mode="rgb", class_names=class_names, follow_links=True)
split_every = int(1 / VAL_RATIO)
train_ds = all_ds.enumerate().filter(lambda index, sample: index % split_every != 0).map(lambda index, sample: sample).shuffle(256).batch(BATCH_SIZE)
val_ds = all_ds.enumerate().filter(lambda index, sample: index % split_every == 0).map(lambda index, sample: sample).batch(BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(filtered_test, shuffle=False, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="int", color_mode="rgb", class_names=class_names, follow_links=True)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# 7. Data augmentation
data_augmentation = tf.keras.Sequential([layers.RandomRotation(factor=0.05), layers.RandomTranslation(height_factor=0.05, width_factor=0.05), layers.RandomZoom(height_factor=0.1), layers.RandomContrast(factor=0.2), layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0))], name="data_augmentation")

# 8. Check
print("\nNumber of classes:", NUM_CLASSES)
print("Classes:", class_names)
# 2. Build ResNet50 model

def build_model(learning_rate=1e-3, dropout_rate=0.3):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs, outputs, name="resnet50")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# 4. Hyperparameter tuning - Trial 1
learning_rate = 5e-4
dropout_rate = 0.3
model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate)
output_dir = Path("/kaggle/working/tuning/trial_1")
output_dir.mkdir(parents=True, exist_ok=True)
callbacks = [
    ModelCheckpoint(filepath=str(output_dir / "best_model.keras"), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_delta=0.001, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_accuracy", mode="max", min_delta=0.001, patience=6, restore_best_weights=True, verbose=1)
]
history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks, verbose=2)
trial_result = pd.DataFrame([{"trial": 1, "learning_rate": learning_rate, "dropout": dropout_rate, "best_val_accuracy": max(history.history["val_accuracy"]), "best_val_loss": min(history.history["val_loss"]), "epochs": len(history.history["loss"])}])
print(trial_result.to_string(index=False))
trial_result.to_csv(output_dir / "trial_result.csv", index=False)
pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)

# 6. Partial fine-tuning
RUN_NUMBER = 1
checkpoint = "/kaggle/working/tuning/trial_1/best_model.keras"
model = tf.keras.models.load_model(checkpoint)
base_model = model.get_layer("resnet50")
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
trainable_backbone_layers = sum(layer.trainable for layer in base_model.layers)
print("Trainable backbone layers:", trainable_backbone_layers)
print("Total backbone layers:", len(base_model.layers))
learning_rate = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

output_dir_2 = Path("/kaggle/working/tuning/trial_2") if RUN_NUMBER == 1 else Path(f"/kaggle/working/tuning/repeated_trial_2_/run_{RUN_NUMBER}")
output_dir_2.mkdir(parents=True, exist_ok=True)
callbacks_2 = [
    ModelCheckpoint(filepath=str(output_dir_2 / "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=3, min_delta=0.001, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor="val_accuracy", mode="max", min_delta=0, patience=5, restore_best_weights=True, verbose=1)
]
run_train_ds = train_ds if RUN_NUMBER == 1 else train_ds.shuffle(buffer_size=256, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
history_2 = model.fit(run_train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks_2, verbose=2)
trial_2_result = pd.DataFrame([{"trial": 2, "learning_rate": 1e-5, "dropout": 0.3, "unfreeze_layers": 30, "best_val_accuracy": max(history_2.history["val_accuracy"]), "best_val_loss": min(history_2.history["val_loss"]), "epochs": len(history_2.history["loss"])}])
print(trial_2_result.to_string(index=False))
trial_2_result.to_csv(output_dir_2 / "trial_result.csv", index=False)
pd.DataFrame(history_2.history).to_csv(output_dir_2 / "history.csv", index=False)

# Check the best checkpoint on validation and test sets
import json
best_model = tf.keras.models.load_model(str(output_dir_2 / "best_model.keras"))
val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)
print("Validation Accuracy:", round(val_accuracy * 100, 2), "%")
print("Validation Loss:", round(val_loss, 4))
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print("Test Accuracy:", round(test_acc * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))
print("Macro Precision:", round(precision, 4))
print("Macro Recall:", round(recall, 4))
print("Macro F1:", round(f1, 4))

results = pd.DataFrame({"Model": ["ResNet50 Baseline", "ResNet50 Tuned"], "Test Accuracy": [0.8920, test_acc], "Macro Precision": [0.8973, precision], "Macro Recall": [0.8920, recall], "Macro F1": [0.8918, f1], "Test Loss": [0.3352, test_loss]})
print(results.round(4).to_string(index=False))
comparison_path = Path("/kaggle/working/tuning/final_comparison.csv") if RUN_NUMBER == 1 else output_dir_2 / "final_comparison.csv"
results.to_csv(comparison_path, index=False)

metrics = {"model": "resnet50_tuned_shared_trial_1", "repeat": RUN_NUMBER, "accuracy": float(test_acc), "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1), "loss": float(test_loss), "validation_accuracy_of_checkpoint": float(val_accuracy), "validation_loss_of_checkpoint": float(val_loss)}
pd.DataFrame([metrics]).to_csv(output_dir_2 / "test_metrics.csv", index=False)
metadata = {"repeat": RUN_NUMBER, "split_method": "every_fifth_sorted_image", "validation_ratio": VAL_RATIO, "num_classes": NUM_CLASSES, "class_names": list(class_names), "shared_trial_1_checkpoint": checkpoint, "trial_2_checkpoint": str(output_dir_2 / "best_model.keras"), "learning_rate": 1e-5, "maximum_epochs": 15, "unfreeze_last_backbone_layers": 30, "trainable_backbone_layers": trainable_backbone_layers, "epochs_recorded": len(history_2.history["loss"]), "best_validation_epoch": int(np.argmax(history_2.history["val_accuracy"])) + 1}
with open(output_dir_2 / "run_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

# 12. Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format="d")
plt.title("ResNet50 Tuned - Confusion Matrix")
plt.tight_layout()
plt.show()
fig.savefig(output_dir_2 / "confusion_matrix.png", dpi=150)

trial_files = [
    ("Trial 1 — Frozen backbone", Path("/kaggle/working/tuning/trial_1/history.csv")),
    ("Trial 2 — Partial fine-tuning", output_dir_2 / "history.csv"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (name, path) in enumerate(trial_files):
    history = pd.read_csv(path)
    best_epoch = history["val_accuracy"].idxmax() + 1
    epochs = range(1, len(history) + 1)
    print(f"{name}: {len(history)} epochs, best val accuracy {history['val_accuracy'].max():.4%} at epoch {best_epoch}")
    axes[row, 0].plot(epochs, history["accuracy"], label="Train")
    axes[row, 0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[row, 0].set_title(f"{name} — Accuracy")
    axes[row, 0].set_xlabel("Epoch")
    axes[row, 0].set_ylabel("Accuracy")
    axes[row, 0].legend()
    axes[row, 0].grid(alpha=0.3)
    axes[row, 1].plot(epochs, history["loss"], label="Train")
    axes[row, 1].plot(epochs, history["val_loss"], label="Validation")
    axes[row, 1].set_title(f"{name} — Loss")
    axes[row, 1].set_xlabel("Epoch")
    axes[row, 1].set_ylabel("Loss")
    axes[row, 1].legend()
    axes[row, 1].grid(alpha=0.3)

plt.tight_layout()
output_path = Path("/kaggle/working/tuning/resnet50_training_curves.png") if RUN_NUMBER == 1 else output_dir_2 / "resnet50_training_curves.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()
print(output_path)

