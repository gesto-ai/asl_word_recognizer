# Model Serve

## Steps to use the Sign Recognizer script

### Step 1: Get the model weights

```
cd model_serve

scp -r team_046@150.136.219.226:/home/team_046/models ./
```

### Step 2: run `sign_recognizer.py`

```
python sign_recognizer.py /path/to/video.mp4
```

You can download a sample video file from [here](https://discord.com/channels/@me/1017414703133237298/1024513846427262976).


## Steps to run the Streamlit app (connected to the Sign Recognizer)

After you have gotten the model weights from Step 1 of running the sign recognizer script, run:

```
streamlit run app.py
```