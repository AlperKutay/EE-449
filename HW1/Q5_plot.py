from utils import part5Plots
import json
f = open('ResultQ5/Q5_cnn4_third.json')
data1 = json.load(f)
f.close()

part5Plots(data1, save_dir="ResultQ5", filename="Q5_cnn4_third.png")