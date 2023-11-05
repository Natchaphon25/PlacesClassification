# import pickle
import requests
from fastapi import FastAPI, Request
from keras.models import load_model
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  
    allow_methods=['*'],  
    allow_headers=['*']   
)

@app.post("/api/placeclass")
async def read_image(image_data: Request):
    image_dataJson = await image_data.json()
    # url_preProcess = 'http://127.0.0.1:8080/api/preprocess'
    url_preProcess = 'http://172.17.0.2:80/api/preprocess'
    
    file_model_path = "/placeClass/model/my_model.h5"

    
    # with open(os.getcwd()+file_model_path, "rb") as file:
    model = load_model(file_model_path)
    
    class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

    # ทำนายด้วยโมเดลและได้ผลลัพธ์เป็นตัวเลข (index)
    # placesPredictor = load_model.load(file_model_path) # อ่าน model
    
    img = requests.post(url_preProcess, json=image_dataJson)
    img = img.json()['img']
    print(img)

    predictions = model.predict(img)

    # หาดัชนีของคลาสที่มีความน่าจะเป็นสูงที่สุด
    predicted_class_index = np.argmax(predictions)
    # นำดัชนีไปหาชื่อคลาสจาก class_names
    predicted_class_name = class_names[predicted_class_index]
    print(predicted_class_name)
    
    return {'this picture is': predicted_class_name}