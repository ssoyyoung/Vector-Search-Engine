# Flask App
from flask import Flask

# Model
#from vectorAPI.yoloModel.models import YoloV3
#from vectorAPI.baseModel.models import Resnet18

# Run
from vectorAPI.extVector import *

app = Flask(__name__)

#yoloM = YoloV3()
#resnet18 = Resnet18()

@app.route('/')
def checkHealthy():
    return "Healthy!"

@app.route('/createVec')
def createVec():
    extractVecAndSave()
    print(res)
    return "done"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8003, debug=True)