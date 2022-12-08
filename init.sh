#!/bin/bash


echo "Creating Virtual Environment"
sleep 1
python -m venv vtex
source vtex/Scripts/activate

echo "Installing package from transformer/requirements.txt"
echo "This step may take a long time"
sleep 1
pip install -r transformer/requirements.txt

echo "Running program"
sleep 1
python transformer/demo.py