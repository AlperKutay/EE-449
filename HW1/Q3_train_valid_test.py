import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import q3
import utils
import json
#Let cuda works
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
train_set, val_set = train_test_split(train_data, test_size=0.1)

test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
test_set = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False)

# Create the data loaders
val_loader = torch.utils.data.DataLoader(val_set, batch_size=q3.BATCH_SIZE, shuffle=False)

#Once my code has worked, It finishes all models.
models = ['mlp_1', 'mlp_2', 'cnn_3', 'cnn_4', 'cnn_5']
for modelselected in models:

    print(f"Training is started for model {modelselected}")
    # Inıt Required Arrays
    training_loss_total = []
    training_accuracy_total = []
    validation_accuracy_total = []
    validation_loss_total = []
    maxPerformanceTask = 0
    for stepX in range(q3.TRAIN_SIZE):
        print(f"Step {stepX + 1} is started")
        #Init models
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
        else:
            print("Model name is not true.")
        #Inıt arrays for each model
        training_loss = []
        training_accuracy = []
        validation_accuracy = []
        validation_loss = []
        test_accuracy = []
        #Inıt optimizer
        optimizer = torch.optim.Adam(model.parameters())
        # Train the model
        for epoch in range(q3.EPOCH_SIZE):
            print(f"Step {stepX + 1} is started")
            model.train()
            train_total = 0
            train_correct = 0
            train_loss = 0
            #DataLoader
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=q3.BATCH_SIZE, shuffle=True)

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = nn.CrossEntropyLoss()(output, labels.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    model.eval()
                    _, predicted = torch.max(output.data, 1)
                    train_total = labels.size(0)#take label size
                    train_correct = (predicted == labels).sum().item()#Take number of corrections
                    train_loss = loss.item()#Take number of loss
                    training_accuracy.append((train_correct / train_total) * 100)  # Save Training accuracy each 10 step
                    training_loss.append(train_loss)  # Save Training loss each 10 step
                    val_correct = 0
                    val_size = 0
                    #Validation
                    for j, (inputs_val, labels_val) in enumerate(val_loader):
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        val_outputs = model(inputs_val)
                        loss = nn.CrossEntropyLoss()(val_outputs, labels_val)
                        # val_loss += loss.item()
                        _, val_pred = val_outputs.max(1)
                        val_size += labels_val.size(0)
                        val_correct += val_pred.eq(labels_val).sum().item()
                    # validation_loss.append((val_loss / val_size) * 100)
                    validation_accuracy.append((val_correct / val_size) * 100)
            #After each model, add each data to required arrays
            validation_accuracy_total.append(validation_accuracy)
            training_loss_total.append(training_loss)
            training_accuracy_total.append(training_accuracy)
            # Arrays ends with '_total' has 5 members which are for each model
            # Evaluate on test set

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in test_set:
                #Doing same things for testing
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = (correct / total) * 100
            test_accuracy.append(test_acc)
            #Choose maxPerformanceTask
            if test_acc > maxPerformanceTask:
                maxPerformanceTask = test_acc
                #to get weight_best switch model to 'CPU'
                model.to('cpu')
                #The reason of there are two condition is name difference of name of first layers.
                if modelselected == "mlp_1" or modelselected == "mlp_2":
                    weight_best = model.fc1.weight.data.numpy()
                elif modelselected == "cnn_3" or modelselected == "cnn_4" or modelselected == "cnn_5":
                    weight_best = model.conv1.weight.data.numpy()
                #After get weight_best, switch again
                model.to(device)
        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))
    #After Train_Size is finished for each model, take average and save to Json file
    average_train_losses = [sum(sub_list) / len(sub_list) for sub_list in zip(*training_loss_total)]
    average_train_accu = [sum(sub_list) / len(sub_list) for sub_list in zip(*training_accuracy_total)]
    average_valid_accu = [sum(sub_list) / len(sub_list) for sub_list in zip(*validation_accuracy_total)]
    print(f"MaxPerformance: {maxPerformanceTask}")
    # Dictionary for json
    dictionary = {
        'name': modelselected,
        'loss_curve': average_train_losses,
        'train_acc_curve': average_train_accu,
        'val_acc_curve': average_valid_accu,
        'test_acc': maxPerformanceTask,
        'weights': weight_best.tolist(),
    }
    with open("ResultQ3/Json_files/Q3_" + modelselected + ".json", "w") as outfile:
        json.dump(dictionary, outfile)

    utils.visualizeWeights(weight_best, save_dir='ResultQ3/Weights', filename='input_weights_' + modelselected)
