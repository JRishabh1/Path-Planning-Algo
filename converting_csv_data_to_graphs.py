import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file = input('Enter the name of the csv file (exclude .csv): ')
data = pd.read_csv(file + ".csv", header=None)

values = data.iloc[0].values
time = data.iloc[1].values

# Compute prefix sum
prefix_sum = values.cumsum()

# Plot of each point
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(values, marker='o', linestyle=' ')
plt.title('Original Values')
plt.xlabel('Iterations')
plt.ylabel('Length of Branch Added')

# Plot using prefix sum
plt.subplot(2, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(prefix_sum, marker='o', linestyle='-')
plt.title('Cumulative Sum')
plt.xlabel('Iterations')
plt.ylabel('Prefix Sum of Length of Branch Added')

plt.subplot(2, 2, 3)
plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
plt.title('Histogram of Branch Lengths')

plt.xlabel('Branch Length')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.plot(time, prefix_sum, marker='o', linestyle='') 

# Customize the plot
plt.title('Time vs Prefix Sum')
plt.xlabel('Time')
plt.ylabel('Prefix Sum')

plt.tight_layout()
plt.savefig(file + '.png', dpi=300, bbox_inches='tight')  
