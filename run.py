# Flask App
from flask import Flask
from app import router

from config import DataConfig as D

app = Flask(__name__)


@app.route('/')
def checkHealthy():
    return "Healthy!"

@app.route('/createVec')
def createVec():
    router.makeDBindex(D.CATE, D.NUM, D.WRITE_M)
    return "done"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8003, debug=True)