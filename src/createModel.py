"""
Script for training the model to recognize the number of fingers shown in a photo.
Data source: Kaggle Hub ("koryakinp/fingers")
"""

import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import kagglehub

imported = True

path = kagglehub.dataset_download("koryakinp/fingers")

print("Path to dataset files:", path)

trainpath = os.path.join(path, "train")

# zaimportowane
if not imported:
    for file in os.listdir(trainpath):
        name = file.split(".")[0][-2:]
        target_dir = os.path.join("photos", name[0])
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(trainpath + "/" + file, target_dir)

# inspiration https://www.tensorflow.org/tutorials/images/classification

class_names = ['0', '1', '2', '3', '4','5']

# zobrazowanie wszystkich - przykład z każdej klasy
plt.figure(figsize=(10,10))
for i in range(6):
    folder = "photos/"+str(i)
    filename = os.listdir(folder)[0]
    img = Image.open(folder + "/" + filename).convert("RGB")
    plt.subplot(3,2,i+1) # 3 wiersze x 2 kolumny
    plt.xticks([]) # etykiety
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(str(i))
plt.show()

# stworzenie tensors
batch_size = 32 # 32 obrazy w jednym kroku epoki
img_height = 128
img_width = 128
print(batch_size, img_height, img_width)

train_photos = tf.keras.utils.image_dataset_from_directory( # nazwa folderu == etykieta klasy
  "photos",
  validation_split=0.15,
  subset="training",
  seed=0, # za każdym razem podział taki sam
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_photos = tf.keras.utils.image_dataset_from_directory(
  "photos",
  validation_split=0.15,
  subset="validation",
  seed=0,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_photos.class_names
print(class_names)

for image_batch, labels_batch in train_photos:
  print(image_batch.shape) # 32 x 128 x 128 x 3 (bo 3 kanały kolorów RGB)
  print(labels_batch.shape)
  break

num_classes = len(class_names)

# ze skalowaniem wartości pikseli [0,255] -> [0.0, 1.0]
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'), # liczba filtrów, rozmiar filtrów -> 16 filtrów wyjściowych
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.GlobalAveragePooling2D(), # uśrednianie na każdym kanale
  layers.Dense(128, activation='relu'), # każdy ze wszystkimi poprzenimi
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', # dopasowuje się do gradientów
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # wartościami prawd. a nie logity
              metrics=['accuracy'])

model.summary()

epochs=18
history = model.fit(
  train_photos,
  validation_data=val_photos,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# epochs_range = range(epochs)
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

model.save('hand_model.keras')
