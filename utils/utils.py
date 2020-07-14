import base64
import numpy as np
from datetime import datetime

dataFormat = "%Y-%m-%d %H:%M:%S"
float32 = np.dtype('>f4')


def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=float32).tolist()


def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(float32)).decode("utf-8")
    return base64_str


def utc_time():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def save_data(data_list):
    with open("result.txt", "a") as f:
        for data in data_list:
            f.write(data+", ")
        f.write("\n")
    print("saved")