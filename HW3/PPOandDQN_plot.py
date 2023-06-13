import json
import matplotlib.pyplot as plt
dir = "ep_rew_mean/"
# Read the JSON files
with open(dir+'PPO_1.json') as file:
    data1 = json.load(file)
with open(dir+'PPO_2.json') as file:
    data2 = json.load(file)
with open(dir+'PPO_3.json') as file:
    data3= json.load(file)

# Extract the data
x1 = [item[1] for item in data1]
y1 = [item[2] for item in data1]

x2 = [item[1] for item in data2]
y2 = [item[2] for item in data2]

x3 = [item[1] for item in data3]
y3 = [item[2] for item in data3]

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the data on the axis
ax.plot(x1, y1, label='PPO 1')
ax.plot(x2, y2, label='PPO 2')
ax.plot(x3, y3, label='PPO 3')


with open(dir+'DQN_1.json') as file:
    data1 = json.load(file)
with open(dir+'DQN_2.json') as file:
    data2 = json.load(file)
with open(dir+'DQN_3.json') as file:
    data3= json.load(file)

# Extract the data
x1 = [item[1] for item in data1]
y1 = [item[2] for item in data1]

x2 = [item[1] for item in data2]
y2 = [item[2] for item in data2]

x3 = [item[1] for item in data3]
y3 = [item[2] for item in data3]

# Create the figure and axis objects
#fig, ax = plt.subplots()

# Plot the data on the axis
ax.plot(x1, y1, label='DQN 1')
ax.plot(x2, y2, label='DQN 2')
ax.plot(x3, y3, label='DQN 3')

# Set the labels and title
ax.set_xlabel('Step Number')
ax.set_ylabel('Value')
ax.set_title('ep_rew_mean Data Comparison')

# Display the legend
ax.legend()
plt.savefig(dir+" PPO and DQN.png")
# Show the plot
plt.show()