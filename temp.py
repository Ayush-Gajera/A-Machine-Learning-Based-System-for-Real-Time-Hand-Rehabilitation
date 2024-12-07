import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming you have the same dataset as before
data = {
    'Estimators': [50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 
                   50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200,
                   50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200,
                   50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200],
    'Test Size': [20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 
                 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0,
                 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0,
                 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0, 20.0, 25.0, 30.0],
    'Split State': [42]*72,
    'RF State': [22]*24 + [42]*24 + [32]*24,
    'Accuracy': [98.21, 98.13, 98.04, 98.66, 98.21, 97.98, 98.41, 98.21, 98.11, 98.31, 98.25, 98.01, 98.31, 98.29, 98.04, 98.41, 98.33, 98.21, 
                 98.11, 98.21, 97.81, 98.41, 98.17, 97.98, 98.51, 98.41, 98.08, 98.56, 98.05, 98.04, 98.61, 98.09, 98.34, 98.66, 98.13, 98.28,
                 98.01, 97.97, 97.94, 98.21, 98.05, 98.11, 98.46, 98.21, 98.21, 98.06, 97.97, 98.04, 98.41, 98.25, 98.28, 98.51, 98.21, 98.41,
                 98.51, 98.13, 98.21, 98.56, 98.33, 98.28, 98.61, 98.53, 98.34, 98.26, 98.29, 98.08, 98.36, 98.49, 98.11, 98.41, 98.41, 98.21]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot 2: Accuracy vs RF State
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='RF State', y='Accuracy', hue='Test Size', style='Test Size', markers=True, palette='tab10')

plt.title('Accuracy vs Random Forest State with Different Test Sizes')
plt.xlabel('Random Forest State')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend(title="Test Size")
plt.show()

import matplotlib.pyplot as plt

# Data lists for parameters and accuracy
estimators = [50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200, 50, 50, 50, 100, 100, 100, 200, 200, 200]
test_size = [20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30]
split_state = [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 32, 32, 32, 32, 32, 32, 32, 32, 32, 40, 40, 40, 40, 40, 40, 40, 40, 40, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
rf_state = [22, 22, 22, 22, 22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 42, 42, 42, 42, 32, 32, 32, 32, 32, 32, 32, 32, 32, 22, 22, 22, 22, 22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
accuracy = [98.21, 98.13, 98.04, 98.66, 98.21, 97.98, 98.41, 98.21, 98.11, 98.31, 98.25, 98.01, 98.31, 98.29, 98.04, 98.41, 98.33, 98.21, 98.11, 98.21, 97.81, 98.41, 98.17, 97.98, 98.51, 98.41, 98.08, 98.56, 98.05, 98.04, 98.61, 98.09, 98.34, 98.66, 98.13, 98.28, 98.01, 97.97, 97.94, 98.21, 98.05, 98.11, 98.46, 98.21, 98.21, 98.06, 97.97, 98.04, 98.41, 98.25, 98.28, 98.51, 98.21, 98.41, 98.51, 98.13, 98.21, 98.26, 98.21, 98.33, 98.36, 98.57, 98.41, 98.08]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(accuracy, label='Accuracy', marker='o', color='b')

# Adding axis labels and title
plt.title('Random Forest Accuracy with Varying Parameters')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')

# Adding legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Add seaborn import

# Your provided data
split_state = [22, 22, 22, 22, 22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 42, 42, 42, 42, 32, 32, 32, 32, 32, 32, 32, 32, 32, 22, 22, 22, 22, 22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
accuracy = [98.21, 98.13, 98.04, 98.66, 98.21, 97.98, 98.41, 98.21, 98.11, 98.31, 98.25, 98.01, 98.31, 98.29, 98.04, 98.41, 98.33, 98.21, 98.11, 98.21, 97.81, 98.41, 98.17, 97.98, 98.51, 98.41, 98.08, 98.56, 98.05, 98.04, 98.61, 98.09, 98.34, 98.66, 98.13, 98.28, 98.01, 97.97, 97.94, 98.21, 98.05, 98.11, 98.46, 98.21, 98.21, 98.06, 97.97, 98.04, 98.41, 98.25, 98.28, 98.51, 98.21, 98.41, 98.51, 98.13, 98.21, 98.26, 98.21, 98.33, 98.36, 98.57, 98.41, 98.08]
test_size = [20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30, 20, 25, 30]

# Ensure all lists are of the same length
min_length = min(len(split_state), len(accuracy), len(test_size))

# Trim lists to the same length
split_state = split_state[:min_length]
accuracy = accuracy[:min_length]
test_size = test_size[:min_length]

# Create a DataFrame for seaborn to use
import pandas as pd
df = pd.DataFrame({
    'split_state': split_state,
    'accuracy': accuracy,
    'test_size': test_size
})

# Plotting accuracy vs split_state with different colors based on test_size
plt.figure(figsize=(10, 6))
sns.lineplot(x='split_state', y='accuracy', hue='test_size', style='test_size', markers=True, palette='tab10', data=df)

# Adding axis labels and title
plt.title('Accuracy vs Split State with Different Test Sizes')
plt.xlabel('Split State')
plt.ylabel('Accuracy (%)')

# Adding legend
plt.legend(title='Test Size')

# Display the plot
plt.grid(True)
plt.show()
