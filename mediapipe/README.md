# Mediapipe
Mediapipe is used for airdraw mathematical expressions in real-time
## Demo
### Demo1
Demo code from Mediapipe's offical website that marks each hand's 21 features in real-time
```
python Demo1.py
```
### Demo2
Code modified from an existant user that counts how many fingers are raised in real-time
```
python Demo2.py
```
## Air Drawing
Draw at the tip of your right index finger when your right finger is the only one raised
- five fingers raised on the left hand: clear drawing 
- five fingers raised on the right hand: take a screenshot and store it in /mediapipe/screenshots
```
python airdraw.py
```