from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import keras
import keras_cv

from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.title="API de prueba"
app.version="0.0.1"
origins = ["*"]
clases=["buena","mala","mixta"]
model_aux=None
model=None
app.add_middleware(
    CORSMiddleware,
     allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def select_model(model_id):
    if model_id==1:
        model = keras.saving.load_model("models/modelo_dark.keras")
    elif model_id==2:
        model = keras.saving.load_model("models/modelo_moblie.keras")
    elif model_id==3:
        model = keras.saving.load_model("models/modelo_resnet.keras")
    return model
@app.post("/analyze/{model_id:int}")
async def analyze(model_id: int, file: UploadFile = File(...)):    
    image = await file.read()
    return predecir(image,model_id)

def predecir(imagen,model_id):
    global model_aux, model

    if(model_aux!=model_id):
        model_aux=model_id
        model=select_model(model_id)
    model = select_model(model_id)
    image = Image.open(BytesIO(imagen))
    image = image.resize((96, 96))
    
    # Convertir la imagen a un array
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch

    # Predict the class of the image
    predictions = model.predict(input_arr)
    print(predictions)
    print(np.argmax(predictions[0]))
    return f'Clase predicha: {clases[np.argmax(predictions[0])]}'

