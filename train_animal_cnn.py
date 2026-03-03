"""
Animal-10 CNN Training Script
Trains a custom CNN on the Animals-10 Kaggle dataset to classify 10 animal species.

Dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10
Usage:
    1. Download the dataset from Kaggle
    2. Extract to a folder (e.g. 'raw-img/')
    3. Run: python train_animal_cnn.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    exit(1)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_DIR = "raw-img"        # Path to Animals-10 dataset
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 30
MODEL_SAVE  = "animal10.h5"
CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog",
               "elephant", "horse", "sheep", "spider", "squirrel"]

# ─────────────────────────────────────────────
#  DATA GENERATORS
# ─────────────────────────────────────────────
def build_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_data = train_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_NAMES
    )

    val_data = train_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_NAMES
    )

    return train_data, val_data

# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(num_classes=10):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train():
    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found at '{DATASET_DIR}'")
        print("Please download Animals-10 from:")
        print("https://www.kaggle.com/datasets/alessiocorrado99/animals10")
        print("and extract to 'raw-img/' folder.\n")
        return

    print("\n[INFO] Building data generators...")
    train_data, val_data = build_generators()

    print(f"[INFO] Classes found: {train_data.class_indices}")
    print(f"[INFO] Train samples: {train_data.samples}")
    print(f"[INFO] Val samples:   {val_data.samples}")

    print("\n[INFO] Building model...")
    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print(f"\n[INFO] Training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ── Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("\n[OK] Training complete!")
    print(f"[OK] Model saved to: {MODEL_SAVE}")
    print("[OK] Plot saved to:  training_results.png")

# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
def test_model(image_path):
    """Quick test of saved model on a single image."""
    if not os.path.exists(MODEL_SAVE):
        print(f"[ERROR] Model file not found: {MODEL_SAVE}")
        return

    import tensorflow as tf
    from tensorflow.keras.preprocessing import image

    model = tf.keras.models.load_model(MODEL_SAVE)
    img = image.load_img(image_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top_idx = np.argmax(preds)
    print(f"\nPrediction: {CLASS_NAMES[top_idx]} ({preds[top_idx]*100:.1f}%)")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(preds[i] * 30)
        print(f"  {name:12s}: {bar} {preds[i]*100:.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_model(sys.argv[1])
    else:
        train()
