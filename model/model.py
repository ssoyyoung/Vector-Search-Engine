from pydantic import BaseModel

class Vector(BaseModel):
    img_b64: str #이미지
    sex: str = None #성별
    type: str = 'mac' #벡터추출방식 (mac, spoc, cat, none)
    k: int = 4 #k갯수
    model: str = 'yolo' #벡터추출모델
    state: str = 'true' #초기 상태값 STATUS

class DoubleIMG(BaseModel):
    img_b64: str #이미지
    c_key: str
    img_url: str