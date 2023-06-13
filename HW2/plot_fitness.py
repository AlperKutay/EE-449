import os
import matplotlib.pyplot as plt
import json
import numpy as np


x_list_10000 = []
x_list_1000 = []
for i in range(9000):
    x_list_10000.append(i)
for i in range(10000):
    x_list_1000.append(i)
    
def plot_fitness_10000(file_name,folder):
    with open(file_name+'.json', 'r') as f:
        data = json.load(f)
    y = list(data.values())
    plt.figure()
    plt.plot(x_list_10000, y[0][1000:10000])
    
    # Customize the plot as desired
    plt.xlabel('Number of generation')
    plt.ylabel('Fitness Values')
    plt.title(folder+"/Iterations: 1000 - 10000")

    # Save the plot as a PNG file
    plt.savefig(file_name+'1000-10000.png')
    plt.close()
    f.close()
    
def plot_fitness_1000(file_name,folder):
    with open(file_name+'.json', 'r') as f:
        data = json.load(f)
    y = list(data.values())
    plt.figure()
    plt.plot(x_list_1000, y[0][0:10000])
    
    # Customize the plot as desired
    plt.xlabel('Number of generation')
    plt.ylabel('Fitness Values')
    plt.title(folder+"/Iterations: 1 - 10000")

    # Save the plot as a PNG file
    plt.savefig(file_name+'1-10000.png')
    plt.close()
    f.close()
    
folder_names = ["num_genes/","num_inds/","parents/","tm_size/","mut_type/","mut_prob/","elites/","discussion/"]
for i in folder_names:
    folders = [f for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))]
    for folder in folders:
        plot_fitness_1000(str(i)+str(folder)+"/fitnes_values",str(i)+str(folder))
        plot_fitness_10000(str(i)+str(folder)+"/fitnes_values",str(i)+str(folder))
        #os.remove(str(i)+str(folder)+"/fitnes_values0-10000.png")
