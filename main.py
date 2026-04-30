from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Meowth Guard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def readRoot():
    return {"message": "Meowth Guard AI is online"}

@app.post("/pokeScan/")
async def scanCard(file: UploadFile = File(...)):

    # imageBytes = await file.read()
    # image = Image.open(io.BytesIO(imageBytes))
    # prediction = model.predict(image)

    return {
        "status": "success",
        "cardName": "Charizard - Base Set",
        "prediction": "Real",
        "confidenceScore": 98.5,
        "reasoning": "blahblah...etc"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)