import torch
import torch.nn as nn 
from torchvision import models, transforms
from PIL import Image
import sys 

dataTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
numFeat = model.fc.in_features
model.fc = nn.Linear(numFeat, 2)

model.load_state_dict(torch.load('meowthGuardV1.pth', weights_only=True))
model.eval()

imagePath = sys.argv[1]
try:
    image = Image.open(imagePath).convert('RGB')
    imageTensor = dataTransforms(image).unsqueeze(0)

    with torch.no_grad():
        output = model(imageTensor)
        probs = torch.nn.functional.softmax(output[0], dim = 0)

        fakeCon = probs[0].item() * 100
        realCon = probs[1].item() * 100

        print(f"\nScanning: {imagePath}") 
        print("-" *30)
        print(f"Probability that it is fake is: {fakeCon:.2f}%")
        print(f"Probability that it is real is: {realCon:.2f}%")

        if fakeCon > realCon:
            print("\n Verdit is that this is a fake card")
        else:
            print("\n Verdit is that this is a real card")

except Exception as e:
    print(f"Error loading the image: {e}")