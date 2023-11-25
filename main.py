from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from NER import nerTask
import numpy as np
import json



app = FastAPI()

class Item(BaseModel):
    body: str

@app.post("/recognise")
async def predict(item: Item):
    result = nerTask(item.body)
    words = [item['word'] for item in result]
    str_words=" ".join(words)
    return json.dumps({"prediction": str_words})

