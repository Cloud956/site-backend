import cv2_rgb as cv2
import numpy as np
from ImageOperations import resizing
import base64
from PIL import Image
import io

file_path = "bibi.jpg"


def load_bibi():
    main_array = cv2.imread(file_path)
    main_array = cv2.cvtColor(main_array, cv2.COLOR_BGR2RGB)
    main_array = resizing(main_array, 1080, 720)
    return main_array


def save_image(image, name):
    cv2.imwrite(name, image)


def image_array_to_bytes(array: np.ndarray) -> bytes:
    into_list = array.tolist()
    str_version = str(into_list)
    as_bite = bytes(str_version, "utf8")
    return as_bite


def bytes_to_image_array(bytes: bytes) -> np.ndarray:
    decoded = bytes.decode(encoding="utf-8")
    back_to_array = eval(decoded)
    as_nd_array = np.array(back_to_array)
    return as_nd_array


def base64_to_image(transfer: str) -> np.ndarray:
    imgdata = base64.b64decode(transfer)
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)


def image_to_base64(image: np.ndarray) -> str:
    img_pil = Image.fromarray(image)
    with io.BytesIO() as output:
        img_pil.save(output, format="PNG")
        base64_image = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_image
