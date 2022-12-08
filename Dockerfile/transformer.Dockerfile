FROM jjanzic/docker-python3-opencv as python
WORKDIR /transformer
RUN mkdir transformer
COPY ./transformer/requirements.txt ./transformer
RUN pip install -r ./transformer/requirements.txt

RUN pip install mediapipe opencv-python
COPY ./transformer ./transformer

# Copy the Model one level up
COPY checkpoints_bttr_data500 .

# Unpack stuff
RUN ["unzip", "transformer/data2.zip", "-d", "./transformer"] 

CMD ["python3", "-u", "transformer/demo.py"]