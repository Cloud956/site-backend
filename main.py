from fastapi import FastAPI
from pydantic import BaseModel

class TransformationRequest(BaseModel):
    name:  str
    image: str
app = FastAPI()
@app.post('/')
async def root(transformation : TransformationRequest):
    return {"message": transformation.name}