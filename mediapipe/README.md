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
- four fingers (no thumb) raised on the left hand: clear drawing 
- four fingers (no thumb) raised on the right hand: take a screenshot and store it in /mediapipe/screenshots

If you want to pause drawing while holding up the right index finger, lift the middle finger up.
For example: If I draw "a b", then I would draw "a" with only right index finger up. Next I lift the middle finger up to pause drawing, and shift my hand over to give some space before drawing "b" (lower middle finger to start drawing again)
```
python airdraw.py
```