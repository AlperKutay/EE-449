from utils import part3Plots

import json

models = ['mlp_1', 'mlp_2', 'cnn_3', 'cnn_4', 'cnn_5']  # Models that will be printed
results = list()
for model in models:
    f = open('ResultQ3/Json_files/Q3_' + model + '.json')


    results.append(json.load(f))
    f.close()
#print results
part3Plots(results, save_dir=r'ResultQ3', filename='part3Plots')
