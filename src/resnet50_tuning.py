import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score)
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_ROOT = Path("/kaggle/input/datasets/lexset/synthetic-asl-alphabet")
OUTPUT_ROOT = Path("/kaggle/working/tuning")
IMG_SIZE, BATCH_SIZE = 224, 32
VAL_RATIO, SPLIT_SEED = 0.20, 42

def load_data():
    train_root = DATA_ROOT / "Train_Alphabet"
    test_root = DATA_ROOT / "Test_Alphabet"
    classes = sorted(p.name for p in train_root.iterdir()
                     if p.is_dir() and p.name not in ("J", "Z"))
    filtered = Path("/kaggle/working/filtered_asl")
    for source, folder_name in ((train_root, "Train_Alphabet"),
                                (test_root, "Test_Alphabet")):
        destination = filtered / folder_name
        destination.mkdir(parents=True, exist_ok=True)
        for name in classes:
            link = destination / name
            if not link.exists():
                os.symlink(source / name, link)

    common = dict(image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                  label_mode="int", color_mode="rgb", class_names=classes,
                  follow_links=True)
    split = dict(validation_split=VAL_RATIO, seed=SPLIT_SEED, shuffle=True)
    from_folder = tf.keras.utils.image_dataset_from_directory
    train = from_folder(filtered / "Train_Alphabet", subset="training", **split, **common)
    val = from_folder(filtered / "Train_Alphabet", subset="validation", **split, **common)
    test = from_folder(filtered / "Test_Alphabet", shuffle=False, **common)
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)
    print("\nNumber of classes:", len(classes))
    print("Classes:", classes)
    return train, val, test, classes


def make_augmentation():
    return tf.keras.Sequential([
        layers.RandomRotation(factor=0.05),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomZoom(height_factor=0.1),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0)),
    ], name="data_augmentation")

def build_model(augmentation, num_classes, dropout_rate=0.3):
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="resnet50")

def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def make_callbacks(checkpoint, stage):
    first_stage = stage == 1
    return [
        ModelCheckpoint(filepath=str(checkpoint), monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                          patience=4 if first_stage else 3, min_delta=0.001,
                          min_lr=1e-6 if first_stage else 1e-7, verbose=1),
        EarlyStopping(monitor="val_accuracy", mode="max",
                      min_delta=0.001 if first_stage else 0,
                      patience=6 if first_stage else 5,
                      restore_best_weights=True, verbose=1),
    ]

def unfreeze_backbone(model):
    base = model.get_layer("resnet50")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    count = sum(layer.trainable for layer in base.layers)
    print("Trainable backbone layers:", count)
    print("Total backbone layers:", len(base.layers))
    return count

def save_history(history, directory):
    pd.DataFrame(history.history).to_csv(directory / "history.csv", index=False)

def run_trial_1(train, val, augmentation, num_classes):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = build_model(augmentation, num_classes, dropout_rate=0.3)
    compile_model(model, learning_rate=5e-4)
    directory = OUTPUT_ROOT / "trial_1"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best_model.keras"
    history = model.fit(train, validation_data=val, epochs=100,
                        callbacks=make_callbacks(checkpoint, stage=1), verbose=2)
    result = pd.DataFrame([{
        "trial": 1, "learning_rate": 5e-4, "dropout": 0.3,
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "best_val_loss": min(history.history["val_loss"]),
        "epochs": len(history.history["loss"]),
    }])
    print(result.to_string(index=False))
    result.to_csv(directory / "trial_result.csv", index=False)
    save_history(history, directory)
    return checkpoint

def run_trial_2(train, val, trial_1_checkpoint):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(str(trial_1_checkpoint))
    unfreeze_backbone(model)
    compile_model(model, learning_rate=1e-5)
    directory = OUTPUT_ROOT / "trial_2"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best_model.keras"
    history = model.fit(train, validation_data=val, epochs=15,
                        callbacks=make_callbacks(checkpoint, stage=2), verbose=2)
    result = pd.DataFrame([{
        "trial": 2, "learning_rate": 1e-5, "dropout": 0.3,
        "unfreeze_layers": 30,
        "best_val_accuracy": max(history.history["val_accuracy"]),
        "best_val_loss": min(history.history["val_loss"]),
        "epochs": len(history.history["loss"]),
    }])
    print(result.to_string(index=False))
    result.to_csv(directory / "trial_result.csv", index=False)
    save_history(history, directory)
    return checkpoint

def evaluate_tuned_resnet50(val_ds, test_ds, checkpoint_path):
    best_model = tf.keras.models.load_model(str(checkpoint_path))
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

    results = pd.DataFrame({
        "Model": ["ResNet50 Baseline", "ResNet50 Tuned"],
        "Test Accuracy": [0.8920, test_acc],
        "Macro Precision": [0.8973, precision],
        "Macro Recall": [0.8920, recall],
        "Macro F1": [0.8918, f1],
        "Test Loss": [0.3352, test_loss],
    })
    print(results.round(4).to_string(index=False))
    results.to_csv(OUTPUT_ROOT / "final_comparison.csv", index=False)
    return y_true, y_pred

def save_confusion_matrix(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format="d")
    plt.title("ResNet50 Tuned - Confusion Matrix")
    plt.tight_layout()
    plt.show()
    fig.savefig(OUTPUT_ROOT / "confusion_matrix.png", dpi=150)

def plot_training_curves():
    trial_files = {
        "Trial 1 — Frozen backbone": OUTPUT_ROOT / "trial_1/history.csv",
        "Trial 2 — Partial fine-tuning": OUTPUT_ROOT / "trial_2/history.csv",
    }
    histories = {}

    for name, path in trial_files.items():
        history_df = pd.read_csv(path)
        histories[name] = history_df
        best_epoch = history_df["val_accuracy"].idxmax() + 1
        print(
            f"{name}: {len(history_df)} epochs; "
            f"best val accuracy = {history_df['val_accuracy'].max():.4%} "
            f"at epoch {best_epoch}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (name, history_df) in enumerate(histories.items()):
        epochs = range(1, len(history_df) + 1)
        best_epoch = history_df["val_accuracy"].idxmax() + 1
        ax_acc, ax_loss = axes[row]

        ax_acc.plot(epochs, history_df["accuracy"], label="Train accuracy", linewidth=2)
        ax_acc.plot(epochs, history_df["val_accuracy"], label="Validation accuracy", linewidth=2)
        ax_acc.axvline(best_epoch, color="gray", linestyle=":", alpha=0.8)
        ax_acc.set(title=f"{name} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.legend()
        ax_acc.grid(alpha=0.25)

        ax_loss.plot(epochs, history_df["loss"], label="Train loss", linewidth=2)
        ax_loss.plot(epochs, history_df["val_loss"], label="Validation loss", linewidth=2)
        ax_loss.axvline(best_epoch, color="gray", linestyle=":", alpha=0.8)
        ax_loss.set(title=f"{name} — Loss", xlabel="Epoch", ylabel="Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.25)

    fig.suptitle("ResNet50 tuning — Training curves", fontsize=15)
    fig.tight_layout()
    output_path = OUTPUT_ROOT / "resnet50_training_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {output_path}")

def repeat_trial_2(train_ds, val_ds, test_ds, class_names, trial_1_checkpoint):
    output_dir = OUTPUT_ROOT / "repeated_trial_2_"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_seeds = {2: 43, 3: 44}
    for run_number, run_seed in run_seeds.items():
        run_dir = output_dir / f"run_{run_number}"
        run_dir.mkdir(parents=True)
        checkpoint_path = run_dir / "best_model.keras"

        print(f"\n{'=' * 60}\nTRIAL 2 REPEAT {run_number} — SEED {run_seed}\n{'=' * 60}")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(run_seed)
        model = tf.keras.models.load_model(str(trial_1_checkpoint))
        trainable_backbone_layers = unfreeze_backbone(model)
        compile_model(model, learning_rate=1e-5)

        run_train_ds = train_ds.shuffle(
            buffer_size=256,
            seed=run_seed,
            reshuffle_each_iteration=True,
        ).prefetch(tf.data.AUTOTUNE)
        history = model.fit(
            run_train_ds,
            validation_data=val_ds,
            epochs=15,
            callbacks=make_callbacks(checkpoint_path, stage=2),
            verbose=2,
        )
        save_history(history, run_dir)
        del model
        gc.collect()
        tf.keras.backend.clear_session()

        best_model = tf.keras.models.load_model(str(checkpoint_path))
        val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)
        test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
        y_true = []
        y_pred = []
        for images, labels in test_ds:
            predictions = best_model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics = {
            "model": "resnet50_tuned_shared_trial_1",
            "repeat": run_number,
            "seed": run_seed,
            "accuracy": float(test_accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "loss": float(test_loss),
            "validation_accuracy_of_checkpoint": float(val_accuracy),
            "validation_loss_of_checkpoint": float(val_loss),
        }
        pd.DataFrame([metrics]).to_csv(run_dir / "test_metrics.csv", index=False)
        metadata = {
            "repeat": run_number,
            "seed": run_seed,
            "split_seed": SPLIT_SEED,
            "validation_ratio": VAL_RATIO,
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "shared_trial_1_checkpoint": str(trial_1_checkpoint),
            "trial_2_checkpoint": str(checkpoint_path),
            "learning_rate": 1e-5,
            "maximum_epochs": 15,
            "unfreeze_last_backbone_layers": 30,
            "trainable_backbone_layers": trainable_backbone_layers,
            "epochs_recorded": len(history.history["loss"]),
            "best_validation_epoch": int(np.argmax(history.history["val_accuracy"])) + 1,
        }
        with open(run_dir / "run_metadata.json", "w") as file:
            json.dump(metadata, file, indent=2)

        print(pd.DataFrame([metrics]).to_string(index=False))
        del best_model
        gc.collect()
        tf.keras.backend.clear_session()

    print("\nSaved outputs in:", output_dir)

def main():
    train_ds, val_ds, test_ds, class_names = load_data()
    data_augmentation = make_augmentation()
    trial_1_checkpoint = run_trial_1(
        train_ds, val_ds, data_augmentation, len(class_names)
    )
    trial_2_checkpoint = run_trial_2(train_ds, val_ds, trial_1_checkpoint)
    y_true, y_pred = evaluate_tuned_resnet50(val_ds, test_ds, trial_2_checkpoint)
    save_confusion_matrix(y_true, y_pred, class_names)
    plot_training_curves()
    repeat_trial_2(train_ds, val_ds, test_ds, class_names, trial_1_checkpoint)

if __name__ == "__main__":
    main()
