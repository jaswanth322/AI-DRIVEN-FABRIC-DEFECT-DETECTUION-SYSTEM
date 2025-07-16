import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
# Load and preprocess images
def load_images_from_folder(folder):
 images = []
 labels = []
 for label, defect_type in enumerate(['Defective', 'Non-Defective']):
 defect_folder = os.path.join(folder, defect_type)
 if not os.path.isdir(defect_folder):
 continue
 for filename in os.listdir(defect_folder):
 img_path = os.path.join(defect_folder, filename)
 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 if img is not None:
 img = cv2.resize(img, (128, 128))
 images.append(img)
 labels.append(label)
 return np.array(images), np.array(labels)
dataset_path = '/content/sample_data/FabricData'
images, labels = load_images_from_folder(dataset_path)
if len(images) == 0:
 raise ValueError("No images were loaded. Please check your dataset
path and structure.")
# Normalize and reshape images
images = images / 255.0
images = images.reshape(-1, 128, 128, 1)
# Split data
x_train, x_test, y_train, y_test = train_test_split(images, labels,
test_size=0.2, random_state=42)
# Compute class weights
class_weights = class_weight.compute_class_weight('balanced',
classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", class_weights_dict)
# Data Augmentation
datagen = ImageDataGenerator(
 rotation_range=20,
 width_shift_range=0.1,
 height_shift_range=0.1,
 shear_range=0.1,
 zoom_range=0.1,
 horizontal_flip=True,
 fill_mode='nearest'
)
datagen.fit(x_train)
# Model Definition
model = models.Sequential([
 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128,
1)),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(128, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(256, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Flatten(),
 layers.Dropout(0.5),
 layers.Dense(128, activation='relu'),
 layers.Dropout(0.5),
 layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=['accuracy'])
model.summary()
# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
restore_best_weights=True)
history = model.fit(datagen.flow(x_train, y_train, batch_size=8),
 epochs=30,
 validation_data=(x_test, y_test),
 callbacks=[early_stopping],
 class_weight=class_weights_dict)
# Plotting Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
# Test with a new image
new_image_path = '/content/sample_data/not_torn_1.jpg' # Update path
if os.path.exists(new_image_path):
 new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)
 new_img = cv2.resize(new_img, (128, 128))
 new_img = new_img / 255.0
 new_img = new_img.reshape(1, 128, 128, 1)
 prediction = model.predict(new_img)
 confidence = prediction[0][0]
 print(f"Prediction Score: {confidence:.4f}")
 if confidence >= 0.66: # At least 70% confident for Non-Defective
 print('Predicted: Non-Defective Fabric')
 elif confidence <= 0.3: # At least 70% confident for Defective
 print('Predicted: Defective Fabric')
 else:
 print('Prediction is uncertain. Please check the image quality or
retrain the model.')
else:
 print(f"Image not found at {new_image_path}.")
