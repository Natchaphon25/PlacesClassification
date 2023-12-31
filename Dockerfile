FROM python:3.11
 
WORKDIR /placeClass

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /placeClass/requirements.txt

COPY ./app /placeClass/app
COPY ./model /placeClass/model

ENV PYTHONPATH "${PYTHONPATH}:/placeClass"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]