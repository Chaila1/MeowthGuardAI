import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify 
import io 
import random

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
print("The AI is alive and awaiting requests on port 8000")

@app.route('/pokeScan/', methods=['POST'])
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
            final_score = realCon if verdict == "real" else fakeCon

            if verdict == "real":
                if realCon > 90:
                    highReal = [
                        "High match. The patterns and texture map perfectly to authentic Generations pack standards",
                        "Authenticity was verified. The border ratios and font weights match the official Generations dataset",
                        "Likely match. There are no structural deviations that were detected in the baseline parameters for this Generations series card"
                    ]
                    reason = random.choice(highReal)
                else:
                    reason = "This card is likely authentic, although minor pixel noise was detected. Key Generations series features were detected then"
            else:
                if fakeCon > 90:
                    highFake = [
                        "Significant deviation detected. Color saturation and border alignment suggest a non-official print.",
                        "Counterfeit warning. Font weight and layering fail the structural integrity check against the Generations dataset.",
                        "High probability of forgery. Edges and contrast ratios do not map to the authentic baseline."
                    ]
                    reason = random.choice(highFake)
                else:
                    reason = "Inconsistent patterns found. Visual markers do not fully match the known authentic Generations dataset."

            return jsonify({
                "status": "success",
                "cardName": "Generations Series Card",
                "prediction": verdict,
                "confidenceScore": round(final_score, 2),
                "reasoning": reason
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)