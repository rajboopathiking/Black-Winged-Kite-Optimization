# Importing required libraries
import numpy as np
import tensorflow as tf


# BWK optimization algorithm
class BWKOptimizer:
    def __init__(self, bounds, num_particles, num_iterations):
        self.bounds = bounds  # Bounds for hyperparameters
        self.num_particles = num_particles  # Number of particles (candidate solutions)
        self.num_iterations = num_iterations  # Number of iterations
        self.position = np.random.uniform(0, 1, (num_particles, len(bounds)))  # Initialize particles randomly
        self.velocity = np.random.uniform(-0.1, 0.1, (num_particles, len(bounds)))  # Small random velocities
        self.best_position = self.position.copy()
        self.best_score = np.inf * np.ones(num_particles)

    def fitness_function(self, model_params,train_data,test_data,model_func, loss, metrics):
        accuracy = self.train_and_evaluate(model_params, model_func, loss, metrics, train_data, test_data)
        print("Minimize Accuracy :", 1 - accuracy)
        return 1 - accuracy  # Minimizing error

    def optimize(self):
        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                # Scale position to real hyperparameter values
                scaled_position = self.bounds[:, 0] + self.position[i] * (self.bounds[:, 1] - self.bounds[:, 0])
                scaled_position[1:] = np.round(scaled_position[1:])  # Convert integer parameters

                # Evaluate fitness
                fitness = self.fitness_function(scaled_position)
                if fitness < self.best_score[i]:
                    self.best_score[i] = fitness
                    self.best_position[i] = self.position[i]

                # Update velocity and position (simple update rule)
                inertia = 0.5  # Controls impact of previous velocity
                cognitive = np.random.rand() * (self.best_position[i] - self.position[i])  # Particle best influence
                self.velocity[i] = inertia * self.velocity[i] + cognitive

                self.position[i] = np.clip(self.position[i] + self.velocity[i], 0, 1)

        best_particle_idx = np.argmin(self.best_score)
        best_scaled_position = self.bounds[:, 0] + self.best_position[best_particle_idx] * (
                    self.bounds[:, 1] - self.bounds[:, 0])
        best_scaled_position[1:] = np.round(best_scaled_position[1:])  # Convert integer parameters
        return best_scaled_position

    def train_and_evaluate(self, params, model_func, loss, metrics, train_data, test_data):
        params = np.array(params).flatten()  # Ensure it's a 1D array
        learning_rate = float(params[0])
        batch_size = int(params[1])
        epochs = int(params[2])

        model = model_func()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[metrics]
        )

        print("Start Training .......")
        history = model.fit(train_data, batch_size=batch_size, epochs=epochs)
        _, test_acc = model.evaluate(test_data)
        return test_acc
