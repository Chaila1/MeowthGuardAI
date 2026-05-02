import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms  
from torch.utils.data import DataLoader
import os

dataTransforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataDir = 'dataset'
imageDataset = datasets.ImageFolder(os.path.join(dataDir), dataTransforms)

dataLoader = DataLoader(imageDataset, batch_size=4, shuffle=True)

classNames = imageDataset.classes
print(f"Classes were found: {classNames}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

numFeat = model.fc.in_features
model.fc = nn.Linear(numFeat, len(classNames))

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.0001)

epochs = 30

print("\n--- Starting up the AI training ---")
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

    runLoss = 0.0
    correctPredict = 0

    for inputs, labels in dataLoader:
        opt.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = crit(outputs, labels)

        loss.backward()
        opt.step()

        runLoss += loss.item() * inputs.size(0)
        correctPredict += torch.sum(preds == labels.data)

    epochLoss = runLoss / len(imageDataset)
    epochAcc = correctPredict.double() / len(imageDataset)

    print(f'Loss: {epochLoss:.4f} | Accuracy: {epochAcc:.4f}\n')

torch.save(model.state_dict(), 'meowthGuardV1.pth')
print("The intense workout has been completed and the new brain has been saved as 'meowthGuardV1.pth'")