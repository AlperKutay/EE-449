import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import q3
import utils
import json

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

# Create the data loaders
val_loader = torch.utils.data.DataLoader(val_set, batch_size=q3.BATCH_SIZE, shuffle=True)

# Create the model and optimizer

models = 'cnn_4'
lrs = [0.1, 0.01, 0.001]  # Learning Rates
model = q3.cnn_4(1024, 10).to(device)


validation_accuracy = []
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=0.00)  # Setting LR
# Train the model
for epoch in range(q3.EPOCH_SIZE + 15):  # to make Epoch_size=30
    if(epoch == 11):#1000/90 = 11
        optimizer = torch.optim.SGD(model.parameters(), lr=lrs[1], momentum=0.00)  # Setting LR
        print("LR is changed to 0.01 from 0.1")
    print(f"Epoch {epoch + 1}/30")
    train_total = 0
    train_correct = 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=q3.BATCH_SIZE, shuffle=True)

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            model.eval()
            running_loss = loss.item()
            val_generator = torch.utils.data.DataLoader(val_set, batch_size=q3.BATCH_SIZE, shuffle=False)
            val_total = 0
            val_correct = 0
            for j, (inputs_val, labels_val) in enumerate(val_loader):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                val_outputs = model(inputs_val)
                loss = nn.CrossEntropyLoss()(val_outputs, labels_val)
                _, val_pred = val_outputs.max(1)
                val_total += labels_val.size(0)
                val_correct += val_pred.eq(labels_val).sum().item()

            validation_accuracy.append((val_correct / val_total) * 100)
    # Evaluate the model on the validation set
    # Evaluate on test set
    print(f"Validation Accuracy: {sum(validation_accuracy) / len(validation_accuracy)} on Epoch : {epoch}")
dictonary = {
    'name': 'cnn_4',
    'loss_curve_1': validation_accuracy,
    'loss_curve_01': validation_accuracy,
    'loss_curve_001': validation_accuracy,
    'val_acc_curve_1': validation_accuracy,
    'val_acc_curve_01': validation_accuracy,
    'val_acc_curve_001': validation_accuracy,
}
# Recording Results
with open("ResultQ5/Q5_cnn4_second.json", "w") as outfile:
    json.dump(dictonary, outfile)
