
!pip install tensorflow_datasets -q

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

(train_ds_raw, val_ds_raw), ds_info = tfds.load(
    'eurosat/rgb',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Extracting class names
CLASS_NAMES = ds_info.features['label'].names
NUM_CLASSES = len(CLASS_NAMES)

print(f"Number of training samples: {len(train_ds_raw)}")
print(f"Number of validation samples: {len(val_ds_raw)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class Names: {CLASS_NAMES}")


IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess_image(image, label):
    """
    Resizes image to IMG_SIZE x IMG_SIZE and applies MobileNetV2-specific preprocessing.
    """
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

# training and validation pipelines
train_ds = train_ds_raw.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds_raw.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Load base model
base_model = MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# new model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
initial_epochs = 10

# checkpoint callback
checkpoint_path = "eurosat_model_checkpoint.weights.h5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True, 
    monitor='val_accuracy', 
    mode='max',             
    save_best_only=True)  

print("--- Starting initial training... ---")

history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback]
)

# load the best weights saved by the checkpoint
print(f"\nTraining complete. Loading best model weights from {checkpoint_path}...")
model.load_weights(checkpoint_path)
print("Best weights loaded into model.")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

y_true = []
y_pred_probs = []

for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images))

y_pred = np.argmax(y_pred_probs, axis=1)

#Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

#Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Unfreezing base model
base_model.trainable = True

fine_tune_at = 100

# Freezing all layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

model.summary()

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs

print("--- Starting fine-tuning... ---")
history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback] 
)

# loading the best weights again
print(f"\nFine-tuning complete. Loading best model weights from {checkpoint_path}...")
model.load_weights(checkpoint_path)
print("Best fine-tuned weights loaded into model.")

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(initial_epochs - 1, linestyle='--', color='k', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(initial_epochs - 1, linestyle='--', color='k', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

print("--- Generating Evaluation Metrics for Fine-Tuned Model ---")

y_true_ft = []
y_pred_probs_ft = []

for images, labels in val_ds:
    y_true_ft.extend(labels.numpy())
    y_pred_probs_ft.extend(model.predict(images))

y_pred_ft = np.argmax(y_pred_probs_ft, axis=1)

# Classification Report
print("\nClassification Report (Fine-Tuned):")
print(classification_report(y_true_ft, y_pred_ft, target_names=CLASS_NAMES))

# Confusion Matrix
print("\nConfusion Matrix (Fine-Tuned):")
cm_ft = confusion_matrix(y_true_ft, y_pred_ft)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (Fine-Tuned)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

model_filename = 'eurosat_mobilenetv2_finetuned.h5'
print(f"Saving final model to {model_filename}...")
model.save(model_filename)
print("Model saved successfully!")



from tensorflow.keras.preprocessing import image

def predict_land_cover(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    print(f"Predicted Class: {CLASS_NAMES[class_idx]} (Confidence: {confidence:.2f})")

    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {CLASS_NAMES[class_idx]}")
    plt.axis('off')
    plt.show()

# Example
predict_land_cover(model, "Road-construction-phases-a-Before-satellite-image-of-part-of-the-study-area-before_Q320.jpg")


