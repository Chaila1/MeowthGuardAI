import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify 
import io 

app = Flask(__name__)

dataTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Loading the Meowth Guard AI....")
model = models.resnet18(weights=None)
numFeat = model.fc.in_features
model.fc = nn.Linear(numFeat, 2)
model.load_state_dict(torch.load('meowthGuardV1.pth', weights_only=True))
model.eval()
print("The AI is alive and awaiting requests on port 5000")

@app.route('/api/scan', methods=['POST'])
def scanImage():
    if 'file' not in request.files:
        return jsonify({"error": "No image file was provided"}), 400
    
    file = request.files['file']

    try:

        imageBytes = file.read()
        image = Image.open(io.BytesIO(imageBytes)).convert('RGB')

        imageTensor = dataTransforms(image).unsqueeze(0)

        with torch.no_grad():
            output = model(imageTensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)

            fakeCon = probs[0].item() * 100
            realCon = probs[1].item() * 100

            verdict = "fake" if fakeCon > realCon else "real"

            return jsonify({
                "verdict": verdict,
                "conFake": round(fakeCon, 2),
                "conReal": round(realCon, 2)
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)