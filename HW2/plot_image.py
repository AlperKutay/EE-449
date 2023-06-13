import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
    
def plot_images(file_names,folder_name):
  # Generate some random images for demonstration
  images = [cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB) for file_name in file_names]
  # Create a 3x3 grid of subplots
  fig, axs = plt.subplots(3, 3)

  # Iterate over the axes and images, and plot each image on a subplot
  for ax, img in zip(axs.flat, images):
      ax.imshow(img)
      ax.axis('off')

  # Set the title of the figure
  fig.suptitle(folder_name+"/Iterations: 2000 - 10000 ")

  # Show the plot
  plt.savefig(folder_name+"/nine_images.png")
  plt.close()

#plot_images("num_genes/15/")
files= []
folder_names = ["num_genes/","num_inds/","parents/","tm_size/","mut_type/","mut_prob/","elites/","discussion/"]
for i in folder_names:
    folders = [f for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))]
    for folder in folders:
      for count in range(2000,11000,1000):
        folder_name = str(i)+str(folder)
        files.append(folder_name+"/iteration_"+str(count)+'.png')
      plot_images(files,folder_name)
      files=[]

