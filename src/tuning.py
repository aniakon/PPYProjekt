"""
Script for tuning an already created model with some more realistic photos, that I took by myself.
It creates a new model - hand_model_finetuned.keras, used both in try.py and Main.py
"""

from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('hand_model.keras')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy', # dobra dla klasyfikacji wieloklasowej z liczbami całk
    metrics=['accuracy']
)

tuning_data = tf.keras.utils.image_dataset_from_directory(
    "tuning",
    image_size=(128, 128),
    batch_size=8,
    shuffle=True
)
model.fit(tuning_data, epochs=10)

model.save('hand_model_finetuned.keras')