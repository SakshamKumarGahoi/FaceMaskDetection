import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)

dataset_path = 'C:/Users/LENOVO/Documents/Github/FaceMaskDetection/'
img_size = (100, 100)
batch_size = 32
learning_rate = 1e-4
EPOCHS = 15

print("Loading Images")
# Data augmentation

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
)

val_generator = datagen.flow_from_directory(
    'dataset/val',
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=learning_rate)   
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print("Training the model")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)
model.save('face_mask_detection_model.h5')
print("Model saved as face_mask_detection_model.h5")
print("Training completed")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy',marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy',marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss',marker='o')    
plt.plot(history.history['val_loss'], label='Validation Loss',marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print("Plotting completed")