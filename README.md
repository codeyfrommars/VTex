# VTex
ECE379K Computer Vision Project

### Demo

To run the demo, run `bash demo.sh`. It may take a while to create a virtual environment, but the demo should start up automatically.

This application has been tested on Python 3.9. Please use Python 3.9 if you are having trouble running it.

If `bash demo.sh` does not run, then please run the following commands
```
python -m venv vtex
source vtex/Scripts/activate

pip install -r transformer/requirements.txt

python transformer/demo.py
```
### Usage

- Draw with your right hand
- Holding up only your right index finger will draw
- Holding up both your right index and right middle finger will pause drawing
- Holding up four fingers on your right hand will take a screenshot and output the LaTeX code to the bash terminal
- Holding up four fingers on your left hand will clear the screen

### Training

To train the model yourself, unzip data2.zip in `/transformer/`. Then, run `train.py` and `test.py`. You may need to modify the file paths in `transformer/dataset.py`



