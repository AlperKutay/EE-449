import json
import matplotlib.pyplot as plt
dir = "ep_rew_mean/"
# Read the JSON files
with open(dir+'PPO_5million.json') as file:
    data1 = json.load(file)

# Extract the data
x1 = [item[1] for item in data1]
y1 = [item[2] for item in data1]

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the data on the axis
ax.plot(x1, y1, label='PPO 2 5 million')


# Set the labels and title
ax.set_xlabel('Step Number')
ax.set_ylabel('Value')
ax.set_title('ep_rew_mean Data Comparison')

# Display the legend
ax.legend()
plt.savefig(dir+"5million.png")
# Show the plot
plt.show()
