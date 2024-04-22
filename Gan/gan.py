import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np

# Define the generator
def build_generator(latent_dim):
    input_layer = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 256, activation='relu')(input_layer)
    x = Reshape((4, 4, 256))(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
    generator = Model(input_layer, x)
    return generator

# Define the discriminator
def build_discriminator(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    discriminator = Model(input_layer, output_layer)
    return discriminator

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    return gan

# Define the training loop
def train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim, dataset):
    num_batches = len(dataset) // batch_size
    for epoch in range(epochs):
        for batch in range(num_batches):
            # Sample random points from the latent space
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            # Generate images using the generator
            generated_images = generator.predict(noise)
            # Select a random batch of images from the dataset
            real_images = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]
            # Concatenate real and generated images into a single batch
            x_batch = np.concatenate([real_images, generated_images])
            # Create labels for real and generated images
            y_batch = np.zeros(2 * batch_size)
            y_batch[:batch_size] = 1
            # Train the discriminator
            discriminator_loss = discriminator.train_on_batch(x_batch, y_batch)
            # Train the generator
            noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
            generator_loss = gan.train_on_batch(noise, np.ones(batch_size))
        print(f'Epoch {epoch + 1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')

    # Save the trained models
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')

# Define parameters
latent_dim = 100
epochs = 100
batch_size = 128

# Load and preprocess dataset
(x_train, _), (_, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0

# Build and compile discriminator and generator
discriminator = build_discriminator(input_shape=(32, 32, 3))
generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Train the GAN
train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim, x_train)
