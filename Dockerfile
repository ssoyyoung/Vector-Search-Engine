#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6
from nvidia/cuda:10.0-base

COPY . /fastapi
WORKDIR /fastapi

RUN apt update -y && apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev vim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt install python3.6 && apt install python3-pip -y
RUN ln -s /usr/bin/pip3 /usr/bin/pip && ln -s /usr/bin/python3.6 /usr/bin/python
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "main_3nd:app", "--host", "0.0.0.0", "--port", "8001"]
