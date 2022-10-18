from curses import BUTTON1_CLICKED
import streamlit as st
from predictor_backend import PredictorBackend
from sign_recognizer.data_processing.wlasl_videos import load_rgb_frames_from_video_dataset
import s3fs
import boto3
import os
import sys
import pandas as pd
from urllib.request import urlretrieve

########################################################
# Defining all constants, helpers and setup
########################################################

fs = s3fs.S3FileSystem(anon=False)
AWS_LAMBDA_URL = os.getenv("AWS_LAMBDA_URL")

# Demo constants
DEMO_VIDEO_URL = "https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4"
DEMO_VIDEO_LABEL = "before"
SAMPLE_VIDEO_URLS = ["https://sign-recognizer.s3.amazonaws.com/new-videos/05742.mp4", "https://sign-recognizer.s3.amazonaws.com/new-videos/05740.mp4", "https://sign-recognizer.s3.amazonaws.com/new-videos/05732.mp4"]

# S3 constants
S3_BUCKET_NAME = "sign-recognizer"
S3_UPLOADED_VIDEOS_FOLDER = "new-videos"

# Make sure we can connect to the s3 client
try:
    S3_CLIENT = boto3.client("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
except Exception as e:
    print(f"Couldn't connect to the S3 client :(. Error {e}")
    sys.exit(1)

# New videos uploaded by user - CSV file URL
NEW_VIDEOS_CSV_FILENAME = os.getenv("USER_FEEDBACK_CSV_S3_FILENAME")

# Retrieve file contents from AWS S3 - useful for feedback gathering
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_csv(s3_path):
    return pd.read_csv(s3_path)

# Default messsages to display when asking for user feedback
CORRECT_PREDICTION_DROPDOWN_TEXT = "Predicted word is correct :)"
INCORRECT_PREDICTION_DROPDOWN_TEXT = "Predicted word is incorrect :("

########################################################
# App code
########################################################

st.header("Welcome to Gesto AI")
st.write("Upload any .mp4 video file of a sign or enter a public video URL and get a predicted word as text!")
st.write("The demo video for this app is [05727.mp4](https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4) (prediction = 'before') from the WLASL dataset.")

input_video_url = st.text_input('Please enter a public URL pointing directly to an .mp4 video:')
if st.button('Click here for sample video URLs'):
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05742.mp4", language="html")
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05740.mp4", language="html")
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05732.mp4", language="html")

video_url = None

##############################
# Option 1: Input video file
##############################
uploaded_video = st.file_uploader("Or upload an .mp4 video file (NOTE: This will automatically upload the video to our S3 bucket and generate a URL for it):")

if uploaded_video is not None:
    print(f"Starting process for video file: {uploaded_video.name}")
    # We'll need this path for uploading the video to S3
    video_s3_path = f"{S3_BUCKET_NAME}/{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"    
    
    # Save video to our S3 bucket
    with fs.open(video_s3_path, mode='wb') as f:
        f.write(uploaded_video.read()) 
    st.write(f"Uploaded video to AWS S3!")

    # We need to generate a presigned S3 URL S3 for OpenCV to access the video
    video_url = S3_CLIENT.generate_presigned_url(ClientMethod='get_object', Params={"Bucket": S3_BUCKET_NAME, "Key": f"{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"})
    print(f"Presigned URL for input video: {video_url}")

##############################
# Option 2: Input video URL
##############################
elif input_video_url:
    print(f"Starting process for public video URL: {input_video_url}")
    
    # If the user is inputting our sample video URLs, then we don't need to re-upload them to our S3 and we can just generate the presigned url
    if input_video_url in SAMPLE_VIDEO_URLS:
        video_filename = os.path.basename(input_video_url)
        video_url = S3_CLIENT.generate_presigned_url(ClientMethod='get_object', Params={"Bucket": S3_BUCKET_NAME, "Key": f"{S3_UPLOADED_VIDEOS_FOLDER}/{video_filename}"})
        print(f"Presigned URL for input video URL: {video_url}")
    else:
        # Encode video URL for ease of naming
        encoded_video_name = hash(input_video_url)

        # IMPORTANT NOTE: Currently we're assuming that the video is an .mp4 video file
        # First, we have to retrieve the video from the URL and store it locally
        local_input_video_path = "input_video.mp4"
        urlretrieve(input_video_url, local_input_video_path)

        # Path that we'll upload the video to in S3
        video_s3_path = f"{S3_BUCKET_NAME}/{S3_UPLOADED_VIDEOS_FOLDER}/{encoded_video_name}.mp4"
        with open(local_input_video_path, "rb") as input_videofile:
            # Save video to our S3 bucket
            with fs.open(video_s3_path, mode='wb') as f:
                f.write(input_videofile.read()) 
            st.write(f"Uploaded video to AWS S3!")
        video_url = S3_CLIENT.generate_presigned_url(ClientMethod='get_object', Params={"Bucket": S3_BUCKET_NAME, "Key": f"{S3_UPLOADED_VIDEOS_FOLDER}/{encoded_video_name}.mp4"})
        print(f"Presigned URL for input video URL: {video_url}")

if video_url is not None:
    ##########################
    # Model prediction logic
    ##########################
    st.video(video_url)
    st.write("Loading model...")
    if AWS_LAMBDA_URL is None:
        print("AWS Lambda URL not found. Initializing model with local code...")
        model = PredictorBackend()
    else:
        print("AWS Lambda URL found! Initializing model with predictor backend...")
        model = PredictorBackend(url=AWS_LAMBDA_URL)
    st.write("Getting prediction...")
    prediction = model.run(video_url)
    st.write(f"Prediction: {prediction}")

    # Print the expected label for the demo video
    if video_url == DEMO_VIDEO_URL:
        st.write(f"Expected label for demo: {DEMO_VIDEO_LABEL}")

    ##########################
    # User feedback logic
    ##########################
    correctness_state = st.selectbox('Would you like to submit feedback?',
                ("Select an option.", CORRECT_PREDICTION_DROPDOWN_TEXT, INCORRECT_PREDICTION_DROPDOWN_TEXT))    
    if correctness_state in {CORRECT_PREDICTION_DROPDOWN_TEXT, INCORRECT_PREDICTION_DROPDOWN_TEXT}:
        print("Starting feedback collection process...")
        if correctness_state == INCORRECT_PREDICTION_DROPDOWN_TEXT:
            st.write("Please tell us what the correct word was. Check back soon for an updated model that learns from your feedback!")
            correct_label = st.text_input("Enter word here", "")
            st.write(f"The correct label you entered: '{correct_label}'. Thanks for your input!")
            
        elif correctness_state == CORRECT_PREDICTION_DROPDOWN_TEXT:
            st.write("Thank you! We're always trying to improve our models from user feedback.")
            correct_label = prediction

        # Add the feedback to a CSV file only if we haven't added feedback to that video URL already
        new_videos_csv_s3_path = f"s3://{S3_BUCKET_NAME}/{NEW_VIDEOS_CSV_FILENAME}"
        df = read_csv(new_videos_csv_s3_path)
        if df is not None and video_url not in df["video_s3_url"].values and correct_label:
            print("Adding user feedback to CSV and uploading to S3...")
            new_row = pd.Series({"video_s3_url": video_url, "predicted_label": prediction, "correct_label": correct_label})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            df.to_csv(new_videos_csv_s3_path, index=False)


    
