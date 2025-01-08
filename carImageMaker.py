import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def add_noise(image, noise_level):
    noisy_image = image + noise_level * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    return np.clip(noisy_image, 0.0, 1.0)  # Clip values to [0, 1] range

# preprocess images
def load_images(image_dir, image_size=(64, 64)):
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
    return np.array(images)

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(latent_dim)
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        encoded_features = self.fc(x)
        return encoded_features

class Decoder(tf.keras.Model):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        self.fc = tf.keras.layers.Dense(8 * 8 * 256, activation='relu')
        self.reshape = tf.keras.layers.Reshape((8, 8, 256))
        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')
        self.conv_transpose3 = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, activation='sigmoid', padding='same')
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.fc(inputs)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        return x

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, output_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(output_shape)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Locatin of where imaegs are stored
image_dir = 'F:/cars'

images = load_images(image_dir)
print("Number of images loaded:", len(images))

# Split data
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
print("Number of training images:", len(train_images))
print("Number of validation images:", len(val_images))

# latent dimension and output shape
latent_dim = 128  
output_shape = (64, 64, 3)

autoencoder = Autoencoder(latent_dim, output_shape)

autoencoder.compile(optimizer='adam', loss='mse')

num_epochs = 200  
batch_size = 32
initial_noise_level = 0.0
final_noise_level = 0.2

# Train the autoencoder with increasing noise levels
for epoch in range(num_epochs):
    noise_level = initial_noise_level + (final_noise_level - initial_noise_level) * epoch / num_epochs
    noisy_train_images = np.array([add_noise(image, noise_level) for image in train_images])
    print(f"Epoch {epoch+1}/{num_epochs}")
    autoencoder.fit(noisy_train_images, train_images, batch_size=batch_size, epochs=1)

denoised_images = autoencoder.predict(val_images)

# Location of where images are outputed
output_dir = 'F:/denoised_images'
os.makedirs(output_dir, exist_ok=True)

for i, image in enumerate(denoised_images):
    filename = os.path.join(output_dir, f'denoised_image_{i}.jpg')
    cv2.imwrite(filename, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))

print("Denoised images saved successfully.")