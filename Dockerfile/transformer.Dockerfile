# Build the training and testing
FROM python:3.10 as python
WORKDIR /transformer
RUN mkdir transformer
COPY ./transformer/requirements.txt ./transformer
RUN pip install -r ./transformer/requirements.txt
COPY ./transformer ./transformer

# Copy the Model one level up
COPY checkpoints_bttr_data500 .

# Unpack stuff
RUN ["unzip", "transformer/data2.zip", "-d", "./transformer"] 

CMD ["python3", "-u", "transformer/test.py"]