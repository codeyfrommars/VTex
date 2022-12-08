#!/bin/bash


echo -e "Creating Virtual Environment"
echo -e "This step may take a long time\n\n"
sleep 1
python -m venv vtex
source vtex/Scripts/activate

echo -e "Installing package from transformer/requirements.txt"
echo -e "This step may take a long time\n\n"
sleep 1
pip install -r transformer/requirements.txt

echo -e "Unzip data\n\n"
sleep 1
unzip transformer/data2.zip -d ./transformer

echo -e "Running program\n\n"
sleep 1
python transformer/demo.py