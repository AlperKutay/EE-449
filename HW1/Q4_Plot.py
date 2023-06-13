from utils import part4Plots

import json

models_all = ["mlp_1", "mlp_2", "cnn_3", "cnn_4", "cnn_5"]
models_mlp = ['mlp_1', 'mlp_2']  # Models that will be printed
models_cnn = ['cnn_3', 'cnn_4', 'cnn_5']
results = list()
#plot all models in single graph
for model in models_all:
    f = open("ResultQ4/Q4_" + model + ".json" )

    #assemble of all data in results
    results.append(json.load(f))
    f.close()
#print results
part4Plots(results, save_dir=r"ResultQ4", filename=f"part4Result_all")
results = list()
#plot all mlp models in single graph
for model in models_mlp:
    f = open("ResultQ4/Q4_" + model + ".json" )
    results.append(json.load(f))
    f.close()
part4Plots(results, save_dir=r"ResultQ4", filename=f"part4Result_mlp")
results = list()
#plot all cnn models in single graph
for model in models_cnn:
    f = open("ResultQ4/Q4_" + model + ".json" )
    results.append(json.load(f))
    f.close()
part4Plots(results, save_dir=r"ResultQ4", filename=f"part4Result_cnn")