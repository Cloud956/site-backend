from fastapi import FastAPI
from pydantic import BaseModel
from ImageOperations import *
from testing import base64_to_image, save_image, image_to_base64
from fastapi.middleware.cors import CORSMiddleware


class TransformationRequest(BaseModel):
    image: str
    param1: float | None = None
    param2: float | None = None
    param3: float | None = None


app = FastAPI()
origins = [
    "https://localhost:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transformations/TO_RGB")
async def root(transformation: TransformationRequest):
    beginning = transformation.image.split(",")[0]
    image64 = transformation.image.split(",")[1]
    image = base64_to_image(image64)
    save_image(image, "original.jpg")
    image_transformed = to_transform(image, cv2.COLOR_BGR2RGB)
    save_image(image_transformed, "transformed.jpg")
    byes = image_to_base64(image_transformed)
    return {"image": beginning +','+ byes}
