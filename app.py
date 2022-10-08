import streamlit as st
from dotenv import dotenv_values
from sign_recognizer.predictor_backend import PredictorBackend

config = dotenv_values(".env") 
AWS_LAMBDA_URL = None
if "AWS_LAMBDA_URL" in config:
    AWS_LAMBDA_URL = config["AWS_LAMBDA_URL"]
DEMO_VIDEO_URL = "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"
DEMO_VIDEO_LABEL = "before"

st.header("Welcome to Gesto AI")
st.write("Upload any video of a sign and get a predicted word as text! The demo video for this app is 05727.mp4 from the WLASL dataset.")

uploaded_video = st.file_uploader("Upload a video...")

if uploaded_video is not None:
    # We'll need this path for opening the video with OpenCV
    video_filepath = uploaded_video.name
    print(f"Video filepath: {video_filepath}")

    # Save video to disk
    with open(video_filepath, mode='wb') as f:
        f.write(uploaded_video.read()) 

    # Open video from disk path - technically not needed because we can feed the bytes-like object to st.video
    st_video = open(video_filepath, 'rb')
    video_bytes = st_video.read()
    st.video(video_bytes)
    st.write("Uploaded video and stored to disk!")

    st.write("Initializing model")
    if AWS_LAMBDA_URL is None:
        st.write("AWS Lambda URL not found. Initializing model with local code...")
        model = PredictorBackend()
        video_url = video_filepath
    else:
        st.write("AWS Lambda URL found! Initializing model with predictor backend...")
        model = PredictorBackend(url=AWS_LAMBDA_URL)
        # Since we don't have a custom URL for the uploaded video, we'll temporarily use the demo video 
        st.write("NOTE: Since we don't have a custom URL for the uploaded video, we'll temporarily return the prediction for the demo video instead!")
        video_url = DEMO_VIDEO_URL
    
    st.write("Getting prediction...")
    prediction = model.run(video_url)
    st.write(f"Final prediction: {prediction}")
    st.write(f"Expected label: {DEMO_VIDEO_LABEL}")


    
