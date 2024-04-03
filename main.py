import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Generating data for neural network training
input_data = np.random.rand(100, 2)
output_data = np.random.rand(100, 1)

# Creation and training of a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(input_data, output_data, epochs=100)

# Create fuzzy variables for input and output signals
natural_light = ctrl.Antecedent(np.arange(0, 101, 1), 'natural_light')
user_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'user_demand')
brightness = ctrl.Consequent(np.arange(0, 101, 1), 'brightness')

natural_light['low'] = fuzz.trimf(natural_light.universe, [0, 0, 50])
natural_light['medium'] = fuzz.trimf(natural_light.universe, [0, 50, 100])
natural_light['high'] = fuzz.trimf(natural_light.universe, [50, 100, 100])

user_demand['low'] = fuzz.trimf(user_demand.universe, [0, 0, 50])
user_demand['medium'] = fuzz.trimf(user_demand.universe, [0, 50, 100])
user_demand['high'] = fuzz.trimf(user_demand.universe, [50, 100, 100])

brightness['low'] = fuzz.trimf(brightness.universe, [0, 0, 50])
brightness['medium'] = fuzz.trimf(brightness.universe, [0, 50, 100])
brightness['high'] = fuzz.trimf(brightness.universe, [50, 100, 100])

# Creating rules for a fuzzy controller
rule1 = ctrl.Rule(natural_light['low'] & user_demand['low'], brightness['high'])
rule2 = ctrl.Rule(natural_light['medium'] | user_demand['medium'], brightness['medium'])
rule3 = ctrl.Rule(natural_light['high'] & user_demand['high'], brightness['low'])

control_system = ctrl.ControlSystem([rule1, rule2, rule3])
controller = ctrl.ControlSystemSimulation(control_system)


# Using a neural network to detect input signals and detect output signals using a fuzzy controller
def fuzzy_control(natural_light_level, user_demand_level):
    predicted_brightness = model.predict(np.array([[natural_light_level, user_demand_level]]))[0][0]
    controller.input['natural_light'] = natural_light_level
    controller.input['user_demand'] = user_demand_level
    controller.compute()
    fuzzy_brightness = controller.output['brightness']
    print("Predicted Brightness (Neural Network):", predicted_brightness)
    print("Fuzzy Brightness:", fuzzy_brightness)
    return predicted_brightness, fuzzy_brightness


# Plotting
natural_light_levels = np.linspace(0, 100, 100)
user_demand_levels = np.linspace(0, 100, 100)

predicted_brightness_values = []
fuzzy_brightness_values = []
for natural_light_level, user_demand_level in zip(natural_light_levels, user_demand_levels):
    predicted_brightness, fuzzy_brightness = fuzzy_control(natural_light_level, user_demand_level)
    predicted_brightness_values.append(predicted_brightness)
    fuzzy_brightness_values.append(fuzzy_brightness)

plt.figure(figsize=(10, 6))
plt.plot(natural_light_levels, predicted_brightness_values, label='Predicted Brightness (Neural Network)')
plt.plot(natural_light_levels, fuzzy_brightness_values, label='Fuzzy Brightness')
plt.title('Brightness Control')
plt.xlabel('Natural Light Level')
plt.ylabel('Brightness Level')
plt.legend()
plt.grid(True)
plt.show()
