import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from BWK_optimizer import BWKOptimizer

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define CNN model
def create_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(
    inputs = inputs, outputs = outputs)
    return model

# Define hyperparameter search space
hyperparameter_bounds = np.array([
    [1e-02, 1e-03],  # Learning rate
    [16, 128],       # Batch size
    [5, 50]          # Epochs
])


# Run BWK Optimization
optimizer = BWKOptimizer(hyperparameter_bounds,num_particles=2,num_iterations=2)
optimizer.fitness_function(hyperparameter_bounds,tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32),tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
                           ,create_model,'sparse_categorical_crossentropy','accuracy')
best_hyperparams = optimizer.optimize()

print("\nBest Hyperparameters Found:")
print(f"Learning Rate: {best_hyperparams[0]:.6f}")
print(f"Batch Size: {int(best_hyperparams[1])}")
print(f"Epochs: {int(best_hyperparams[2])}")
