import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import time

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 10
latent_dim = 2
BASE_PATH = r"C:\Users\arshb\OneDrive\Desktop\pattern\autoencoder\genres_original"
epochs = 20

# Function to load audio data
def load(file_):
    data_, _ = librosa.load(file_, sr=3000, offset=0.0, duration=30)
    return data_.reshape(1, 90001).astype(np.float32)

# Map loading function to filenames
map_data = lambda filename: (tf.py_function(load, [filename], [tf.float32]))

# Model definition
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(1, 90001)),
            layers.Conv1D(64, 1, 2, padding="same"),
            layers.Conv1D(128, 1, 2, padding="same"),
            layers.Conv1D(256, 1, 2, padding="same"),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Reshape(target_shape=(1, latent_dim)),
            layers.Conv1DTranspose(512, 1, 1, padding="same"),
            layers.Conv1DTranspose(256, 1, 1, padding="same"),
            layers.Conv1DTranspose(128, 1, 1, padding="same"),
            layers.Conv1DTranspose(64, 1, 1, padding="same"),
            layers.Conv1DTranspose(1, 1, 1, padding="same")
        ])

# Load dataset
def load_dataset(class_):
    music_list = sorted(os.listdir(os.path.join(BASE_PATH, class_)))
    TrackSet = [os.path.join(BASE_PATH, class_, x) for x in music_list]
    return TrackSet

TrackSet = load_dataset('jazz')
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(TrackSet)
    .map(map_data, num_parallel_calls=AUTOTUNE)
    .shuffle(3)
    .batch(BATCH_SIZE)
)

# Training loop
def train(train_dataset, model):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_x = np.asarray(train_x)[0]
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            test_x = np.asarray(test_x)[0]
            loss(compute_loss(model, test_x))
        display.clear_output(wait=False)
        elbo = -loss.result()
        print(f'Epoch: {epoch}, Test set ELBO: {elbo}, Time: {end_time - start_time}')

# Instantiate the model
model = CVAE(latent_dim)

# Start training
train(train_dataset, model)
