#%% Libraries
from tensorneat.problem.func_fit import FuncFit  
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, common
from PIL import Image

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

#%% Load Images

# List to store all loaded images
samples_dir = 'images'
images = [Image.open(os.path.join(samples_dir, file)) for file in os.listdir(samples_dir) 
          if file.startswith(('piskel_'))]

#%% Plot Images
# Display all images in a grid
plt.figure(figsize=(5, 5))

for i, img in enumerate(images):
    plt.subplot(5, 5, i+1)
    plt.imshow(img)
    plt.axis('off')  # Hide axes

plt.tight_layout()
plt.show()


#%% ColorDetection_old Class

# Define a class for color detection that inherits from FuncFit
class ColorDetection_old(FuncFit):
    # Constructor method to initialize the class
    def __init__(self, images, error_method="mse"):
        self.images = images  # Assign input images to an instance attribute
        self.error_method = error_method  # Assign error method (default is "mse")
        self.pixel_data, self.color_labels = self._prepare_data()  # Prepare pixel data and color labels

    # Method to prepare data from the images
    def _prepare_data(self):
        pixel_data = []  # List to hold pixel data
        color_labels = []  # List to hold corresponding color labels
        
        # Loop through each image in the provided images
        for img in self.images:
            img_array = jnp.array(img)  # Convert image to a JAX array
            # Loop through each row in the image array
            for row in img_array:
                # Loop through each pixel in the row
                for pixel in row:
                    # Extract RGB values from the pixel (assumes pixel has RGBA format)
                    r, g, b, _ = pixel[:4]
                    pixel_data.append([r, g, b])  # Append RGB values to pixel_data list

                    # Determine the color category based on RGB values
                    if r > 150 and g < 100 and b < 100:
                        color_labels.append([1, 0, 0, 0])  # Red
                    elif 100 < r < 150 and 50 < g < 100 and b < 50:
                        color_labels.append([0, 1, 0, 0])  # Brown
                    elif g > 150 and r < 100 and b < 100:
                        color_labels.append([0, 0, 1, 0])  # Green
                    elif b > 150 and r < 100 and g < 100:
                        color_labels.append([0, 0, 0, 1])  # Blue
                    else:
                        color_labels.append([0, 0, 0, 0])  # No category

        # Return pixel data and color labels as JAX arrays
        return jnp.array(pixel_data, dtype=float), jnp.array(color_labels, dtype=float)

    # Property to get input data for the NEAT algorithm
    @property
    def inputs(self):
        return self.pixel_data

    # Property to get target data for the NEAT algorithm
    @property
    def targets(self):
        return self.color_labels

    # Property to get the shape of input data
    @property
    def input_shape(self):
        return self.pixel_data.shape

    # Property to get the shape of target data
    @property
    def output_shape(self):
        return self.color_labels.shape


#%% ColorDetection Class New
class ColorDetection(FuncFit):
    def __init__(self, images, error_method="mse"):
        self.images = images
        self.error_method = error_method
        self.pixel_data, self.color_labels = self._prepare_data()

    def _prepare_data(self):
        pixel_data = []
        color_labels = []

        for img in self.images:
            img_array = jnp.array(img)
            height, width, _ = img_array.shape  # Get the dimensions of the image

            for y in range(height):
                for x in range(width):
                    # Extract RGB values and normalize the coordinates
                    r, g, b, _ = img_array[y, x][:4]
                    normalized_x = (2 * x / (width - 1)) - 1  # Normalize x coordinate
                    normalized_y = (2 * y / (height - 1)) - 1  # Normalize y coordinate

                    pixel_data.append([normalized_x, normalized_y, r, g, b])  # Include normalized coords

                    # Assign target color category based on RGB values
                    if r > 150 and g < 100 and b < 100:
                        color_labels.append([1, 0, 0, 0])  # Red
                    elif 100 < r < 150 and 50 < g < 100 and b < 50:
                        color_labels.append([0, 1, 0, 0])  # Brown
                    elif g > 150 and r < 100 and b < 100:
                        color_labels.append([0, 0, 1, 0])  # Green
                    elif b > 150 and r < 100 and g < 100:
                        color_labels.append([0, 0, 0, 1])  # Blue
                    else:
                        color_labels.append([0, 0, 0, 0])  # No category

        return jnp.array(pixel_data, dtype=float), jnp.array(color_labels, dtype=float)

    @property
    def inputs(self):
        return self.pixel_data

    @property
    def targets(self):
        return self.color_labels

    @property
    def input_shape(self):
        return self.pixel_data.shape

    @property
    def output_shape(self):
        return self.color_labels.shape
#%% Exectution of Pipeline

# Algorithm configuration for NEAT
algorithm = algorithm.NEAT(
    pop_size=1000,  # Population size
    species_size=20,  # Size of species
    survival_threshold=0.1,  # Threshold for survival
    genome=genome.DefaultGenome(
        num_inputs=5,  # Normalized Pixel Coordinates and Number of input features (RGB values)
        num_outputs=4,  # Number of output categories (Red, Brown, Green, Blue)
        output_transform=common.ACT.sigmoid,  # Activation function for output layer
    ),
)

# Initialize the ColorDetection problem with your loaded images
problem = ColorDetection(images)

# Set up the NEAT pipeline
pipeline = Pipeline(
    algorithm=algorithm,  # The configured NEAT algorithm
    problem=problem,  # The problem instance
    generation_limit=200,  # Maximum generations to run
    fitness_target=-1e-6,  # Target fitness level
    seed=42,  # Random seed for reproducibility
)

# Run the pipeline setup
state = pipeline.setup()
# Run the NEAT algorithm until termination
state, best = pipeline.auto_run(state)
# Display the results of the pipeline run
pipeline.show(state, best)


# Visualize Network
network = algorithm.genome.network_dict(state, *best)
algorithm.genome.visualize(network, save_path="images/colordection_network.png")