# syntax=docker/dockerfile:1

FROM python:3.8.3
FROM tensorflow/tensorflow:latest

WORKDIR /workdir

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "autotune_keras.py"]