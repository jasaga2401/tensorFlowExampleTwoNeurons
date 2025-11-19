# ------------------------------------------------------------
# TENSORFLOW: Two-Neuron Hidden Layer (same problem)
# ------------------------------------------------------------
# The model learns the single rule:
#       y = 3x + 1
#
# But this time we add a hidden layer with TWO neurons.
# The output is still ONE number.
# ------------------------------------------------------------

import tensorflow as tf
import numpy as np

# 1. Training data
xs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
ys = np.array([1, 4, 7, 10, 13, 16], dtype=float)   # y = 3x + 1

# 2. Build a model with a hidden layer of TWO neurons
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),          # one input x
    tf.keras.layers.Dense(2, activation='relu'),   # hidden layer with TWO neurons
    tf.keras.layers.Dense(1)             # output layer (ONE output y)
])

# 3. Compile
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# 4. Train
model.fit(xs, ys, epochs=500, verbose=False)
print("Training complete!")
print("----------------------------------------")

# 5. Predict for x = 10
x_new = np.array([[10.0]], dtype=float)
prediction = model.predict(x_new)

print("Prediction for x = 10:")
print(prediction)
print("----------------------------------------")

# 6. Examine learned weights
print("MODEL WEIGHTS:")
for layer in model.layers:
    print(layer.get_weights())
