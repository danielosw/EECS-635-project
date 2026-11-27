

import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout,
    GlobalAveragePooling2D, Dense,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[2]

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128


def parse_args():
    parser = argparse.ArgumentParser(description='Train the ASL gesture classifier.')
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data' / 'ASL_Gestures_36_Classes',
                        help='Folder containing train/ and test/ subfolders')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models',
                        help='Where trained models are saved')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_generators(train_dir, test_dir, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
    )
    return train_gen, test_gen


def build_model(num_classes):
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = args.data_dir / 'train'
    test_dir = args.data_dir / 'test'
    for d in (train_dir, test_dir):
        if not d.exists():
            raise SystemExit(
                f'Dataset folder not found: {d}\n'
                'Download the dataset from Kaggle, then either place it at '
                f'{ROOT / "data" / "ASL_Gestures_36_Classes"} or pass --data-dir.'
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_generator, test_generator = build_generators(train_dir, test_dir, args.batch_size)
    model = build_model(train_generator.num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(str(args.output_dir / 'asl_model_boosted.keras'),
                        monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ]

    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output_dir / 'asl_final_boosted_model.keras')
    print('Training Complete! ✅')

if __name__ == '__main__':
    main()