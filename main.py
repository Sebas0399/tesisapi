from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import onnxruntime as rt
from onnxruntime.datasets import get_example
from onnx_tf.backend import prepare
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.title="API de prueba"
app.version="0.0.1"
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    image = await file.read()
    return predecir(image)

model = tf.keras.models.load_model("models/model.h5")
sess = rt.InferenceSession('best.onnx', providers=rt.get_available_providers())
clases=["buena","mixta","mala"]
input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)
def predecir(imagen):
    image = Image.open(BytesIO(imagen))
    image = image.resize((224, 224))
    # Convertir la imagen a un array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Añadir una dimensión extra
    img_array = tf.expand_dims(img_array, 0)

    # Predecir la clase de la imagen
    #predictions = model.predict(img_array)

    image = np.array(image).astype(np.float32) / 255.0
    # Añadir dimensión de batch (1, altura, ancho, canales)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
   
    res = sess.run([output_name], {input_name: image})
    #predicted_class = tf.argmax(predictions[0]).numpy()
    
    return f'Clase predicha: {clases[np.argmax(res[0][0])]}'