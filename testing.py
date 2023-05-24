import cv2_rgb as cv2
import numpy as np
from ImageOperations import resizing
file_path = 'bibi.jpg'

def load_bibi():
    main_array = cv2.imread(file_path)
    main_array = cv2.cvtColor(main_array, cv2.COLOR_BGR2RGB)
    main_array = resizing(main_array, 1080, 720)
    return main_array
def image_array_to_bytes(array : np.ndarray) -> bytes:
    into_list = array.tolist()
    str_version = str(into_list)
    as_bite = bytes(str_version, 'utf8')
    return as_bite

def bytes_to_image_array(bytes : bytes) ->np.ndarray:
    decoded = bytes.decode(encoding='utf-8')
    back_to_array = eval(decoded)
    as_nd_array = np.array(back_to_array)
    return as_nd_array

main_array = cv2.imread(file_path)
main_array = cv2.cvtColor(main_array, cv2.COLOR_BGR2RGB)
main_array = resizing(main_array, 1080, 720)
main_array = resizing(main_array,150)



b=2

b=2
