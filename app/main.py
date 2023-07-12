from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from ImageOperations import *
from app.transformations import handle_start, image_to_base64
from fastapi.middleware.cors import CORSMiddleware


class TransformationRequest(BaseModel):
    image: str
    param1: float | None = None
    param2: float | None = None
    param3: float | None = None


app = FastAPI()
origins = [
    "http://18.184.42.144:5173",
    "https://18.184.42.144:5173",
    "http://18.184.42.144:443",
    "https://18.184.42.144:443",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transformations/to_bgr")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = to_transform(image, cv2.COLOR_BGR2RGB)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/to_hsv")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = to_transform(image, cv2.COLOR_RGB2HSV)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/to_gray")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = to_transform(image, cv2.COLOR_RGB2GRAY)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/to_hls")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = to_transform(image, cv2.COLOR_RGB2HLS)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/k_means")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = k_means(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/sobel_edge")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = giveShapes(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/linear_sampling")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = linear_sampling(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/nn_sampling")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = nearest_sampling(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/uniform_quantization")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = uniform_quan(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/gauss_noise")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_noise(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/inverse")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_inverse(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/power_law")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_power_law(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/cartoon")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = cartoonify(image, p1, p2, p3)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/translation")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_translate(image, p1, p2)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/salt_pepper")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_salt_pepper(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/median_filter")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_median_filter(image, p1)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/horizontal_noise")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_periodic_noise_horizontal(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/vertical_noise")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = main_periodic_noise_vertical(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/fft_power")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = givePower(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/fft_magnitude")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = giveMagnitude(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}


@app.post("/transformations/denoise")
async def root(transformation: TransformationRequest, request: Request):
    header = request.headers.get("Origin")
    if header not in origins:
        raise HTTPException(status_code=422, detail="Unprocessable entity.")
    image = transformation.image
    beginning, image = handle_start(image)
    p1, p2, p3 = transformation.param1, transformation.param2, transformation.param3
    image_transformed = denoise(image)

    byes = image_to_base64(image_transformed)

    return {"image": beginning + "," + byes}
