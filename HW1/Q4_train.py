import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import q3
import utils
import json
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()
print(f"Device is : {device}")
# Define transforms for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # convert PIL image to tensor
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # normalize to [-1, 1]
    transforms.Grayscale()
])

# Load the CIFAR10 dataset
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)


# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=q3.BATCH_SIZE, shuffle=True)

maxPerformanceTask = 0

#this array has size [5][2]. 5 represents types of models and 2 represents types of activation functions
models = (
    ("mlp_1", "mlp_1_sigmoid"), ("mlp_2", "mlp_2_sigmoid"), ("cnn_3", "cnn_3_sigmoid"), ("cnn_4", "cnn_4_sigmoid"),
    ("cnn_5", "cnn_5_sigmoid"))
#firstly we choose represents types of models
for BothModel in models:
    print(BothModel)
    # Inıt arrays for each model
    relu_loss_curve = []
    relu_grad_curve = []
    sigmoid_loss_curve = []
    sigmoid_grad_curve = []
    #then we choose types of activation functions
    for modelselected in BothModel:
        print(f"Training is started for model {modelselected}")
        for stepX in range(1):
            print(f"Step {stepX + 1} is started")
            if modelselected == 'mlp_1':
                model = q3.mlp_1(1024, 32, 10).to(device)
            elif modelselected == 'mlp_2':
                model = q3.mlp_2(1024, 32, 64, 10).to(device)
            elif modelselected == 'cnn_3':
                model = q3.cnn_3(1024, 10).to(device)
            elif modelselected == 'cnn_4':
                model = q3.cnn_4(1024, 10).to(device)
            elif modelselected == 'cnn_5':
                model = q3.cnn_5(1024, 10).to(device)
            elif modelselected == 'mlp_1_sigmoid':
                model = q3.mlp_1_sigmoid(1024, 32, 10).to(device)
            elif modelselected == 'mlp_2_sigmoid':
                model = q3.mlp_2_sigmoid(1024, 32, 64, 10).to(device)
            elif modelselected == 'cnn_3_sigmoid':
                model = q3.cnn_3_sigmoid(1024, 10).to(device)
            elif modelselected == 'cnn_4_sigmoid':
                model = q3.cnn_4_sigmoid(1024, 10).to(device)
            elif modelselected == 'cnn_5_sigmoid':
                model = q3.cnn_5_sigmoid(1024, 10).to(device)
            else:
                print("Model name is not true.")
            # Inıt optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
            # Train the model
            for epoch in range(q3.EPOCH_SIZE):
                print(f"Epoch {epoch + 1}/15")
                model.train()
                train_total = 0
                train_correct = 0
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=q3.BATCH_SIZE, shuffle=True)
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    model.to('cpu')
                    weightBefore = model.fc1.weight.data.numpy().flatten()
                    model.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels.to(device))

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss = loss.item()

                    if (i + 1) % 10 == 0:
                        model.to('cpu')
                        weightAfter = model.fc1.weight.data.numpy().flatten()
                        model.to(device)
                        running_grad = float(np.linalg.norm(weightAfter - weightBefore) / 0.01)

                        if modelselected[-1] == 'd':  # That means it is sigmoid EX:mlp_1_sigmoid
                            sigmoid_loss_curve.append(running_loss)
                            sigmoid_grad_curve.append(running_grad)
                        else:  # That means it is RELU
                            relu_loss_curve.append(running_loss)
                            relu_grad_curve.append(running_grad)

                # Evaluate the model on the validation set
                # Evaluate on test set
    # Dictionary for json
    if modelselected[6] == 's':#In models array sigmoid models are always come after Relu so that when models with Sigmoid funcitons, we can save data and run other model
        dictionary = {
            'name': modelselected[:5],  # mlp_1 mlp_2 cnn_1 etc.
            'relu_loss_curve': relu_loss_curve,  # the training loss curve of the ANN with ReLU
            'sigmoid_loss_curve': sigmoid_loss_curve,  # the training loss curve of the ANN with logistic sigmoid
            'relu_grad_curve': relu_grad_curve,  # the curve of the magnitude of the loss gradient of the ANN with ReLU
            'sigmoid_grad_curve': sigmoid_grad_curve,
            # the curve of the magnitude of the loss gradient of the ANN with ReLU
        }
        # print(train_losses_total)
        with open("resultQ4/Q4_" + modelselected[:5] + ".json", "w") as outfile:
            json.dump(dictionary, outfile)
        print(f"Both Models which are {BothModel[0]} and {BothModel[1]} are finished.")
