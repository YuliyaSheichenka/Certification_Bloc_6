# app/main.py
import mlflow 
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI
from app.db import database, User
import warnings
import platform
from PIL import Image
from app.utils import image2tensor, pred2label, labels
import tensorflow as tf
import numpy as np

app = FastAPI(title="BrainSight Public API")

@app.get("/")
async def read_root():
    return await User.objects.all()

#class PredictionFeatures(BaseModel):
#    BrainDisease: float


@app.post("/predictionalz", tags=["Deep Learning classification: Alzheimer"])
async def post_picture(file: UploadFile= File(...)):
    """ 
    Alzheimer detection
    """
    img = await file.read()
    img_batch = image2tensor(img, dim=(176, 208))
    
    logged_model = 'runs:/b347c773a181434fae3e122921c1d937/model'
    model = mlflow.tensorflow.load_model(logged_model, keras_model_kwargs={'compile':False})
    
    prediction = model.predict(img_batch)[0]    
    print(prediction)
        
    prediction_label = pred2label(prediction)
        
    response = {
            "prediction" : prediction.tolist(), 
            "predicted_label": prediction_label, 
            "labels": labels
            }
    print(response)
    return response



@app.post("/predictionbt", tags=["Deep Learning classification: Brain Tumors"])
async def post_picture(file: UploadFile= File(...)):
    """
    Brain tumor detection 
    """
    img = await file.read()
    img_batch = image2tensor(img, dim=(176, 208))

    logged_model = 'runs:/b347c773a181434fae3e122921c1d937/model'
    model = mlflow.tensorflow.load_model(logged_model, keras_model_kwargs={'compile':False})

    prediction = model.predict(img_batch)[0]
    print(prediction)

    prediction_label = pred2label(prediction)

    response = {
            "prediction" : prediction.tolist(),
            "predicted_label": prediction_label,
            "labels": labels
            }
    print(response)
    return response

@app.on_event("startup")
async def startup():
    if not database.is_connected:
        await database.connect()
    # create a dummy entry
    await User.objects.get_or_create(email="test@test.com")


@app.on_event("shutdown")
async def shutdown():
    if database.is_connected:
        await database.disconnect()
