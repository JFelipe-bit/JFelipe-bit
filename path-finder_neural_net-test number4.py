import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the TIFF image
image_path = "C:\Users\jose.barradas\Desktop\Geoviewer\Images to shape files\outputs"
image = Image.open(image_path)

# Convert the image to RGB mode if not already in RGB
if image.mode != "RGB":
    image = image.convert("RGB")

# Define the target color (replace with desired RGB values)
target_color = (255, 0, 0)  # Red color as an example

# Preprocess the image
data = np.array(image)
normalized_data = data / 255.0  # Normalize pixel values between 0 and 1

# Find the pixels that match the target color using AI
# (neural network could be used here for pattern recognition)
match_pixels = np.all(normalized_data == np.array(target_color) / 255.0, axis=-1)

# Generate input and output data for the neural network
input_data = normalized_data.reshape(-1, 3)
output_data = match_pixels.flatten()

# Create and train the neural network model
# (using a simple neural network architecture)
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(3,)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(input_data, output_data, epochs=10)

# Use the trained model to find the path using AI prediction
predicted_data = model.predict(input_data).flatten()
path_pixels = predicted_data > 0.5

# Generate a new image with the path highlighted
path_image = np.zeros_like(data)
path_image[path_pixels] = data[path_pixels]

# Save the resulting image
path_image = Image.fromarray(path_image, mode="RGB")
path_image.save("C:\Users\jose.barradas\Desktop\Geoviewer\Images to shape files\outputs")
