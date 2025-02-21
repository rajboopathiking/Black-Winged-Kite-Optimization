# Black-Winged-Kite-Optimization

Here’s how to apply Black-Winged Kite Optimization (BWK) for hyperparameter tuning in an classification project using TensorFlow/Keras and the BWK algorithm.

### Steps :
```sh 
git clone https://github.com/rajboopathiking/Black-Winged-Kite-Optimization.git 
```
```python
from BWK_optimizer import BWKOptimizer

optimizer = BWKOptimizer(bounds, num_particles, num_iterations)
optimizer.fitness_function(params, model_func, loss, metrics, train_data, test_data)
optimizer.optimize()
```

### Example Project

##### Check Test.py in Respository
```
hyperparameter_bounds = np.array([
    [1e-02, 1e-03],  # Learning rate
    [16, 128],       # Batch size
    [5, 50]          # Epochs
])
```
