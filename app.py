import streamlit as st
from dotenv import dotenv_values
from predictor_backend import PredictorBackend
import s3fs
import os

fs = s3fs.S3FileSystem(anon=False)
AWS_LAMBDA_URL = os.getenv("AWS_LAMBDA_URL")

# Demo constants
DEMO_VIDEO_URL = "https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4"
DEMO_VIDEO_LABEL = "before"

# S3 constants
S3_BUCKET_NAME = "sign-recognizer"
S3_UPLOADED_VIDEOS_FOLDER = "new-videos"

st.header("Welcome to Gesto AI")
st.write("Upload any video of a sign and get a predicted word as text! The demo video for this app is [05727.mp4](https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4) (prediction = 'before') from the WLASL dataset.")

uploaded_video = st.file_uploader("Upload a video...")

if uploaded_video is not None:
    # We'll need this path for opening the video with OpenCV
    short_s3_video_url = f"{S3_BUCKET_NAME}/{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"    
    full_s3_video_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"
    print(f"Video S3 URL: {full_s3_video_url}")

    # Save video to our S3 bucket
    with fs.open(short_s3_video_url, mode='wb') as f:
        f.write(uploaded_video.read()) 

    # Open video from disk path - technically not needed because we can feed the bytes-like object to st.video
    st.video(full_s3_video_url)
    st.write("Uploaded video and stored to disk!")

    if AWS_LAMBDA_URL is None:
        st.write("AWS Lambda URL not found. Initializing model with local code...")
        model = PredictorBackend()
    else:
        st.write("AWS Lambda URL found! Initializing model with predictor backend...")
        model = PredictorBackend(url=AWS_LAMBDA_URL)
    
    st.write("Getting prediction...")
    prediction = model.run(full_s3_video_url)
    st.write(f"Final prediction: {prediction}")

    # Print the expected label for the demo video
    if full_s3_video_url == DEMO_VIDEO_URL:
        st.write(f"Expected label for demo: {DEMO_VIDEO_LABEL}")


    
