FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

COPY . /Vector-Search-Engine
WORKDIR /Vector-Search-Engine

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8001

#CMD ["uvicorn", "main_2nd:app", "--host", "0.0.0.0", "--port", "8001"]
