import streamlit as st
from dotenv import dotenv_values
from predictor_backend import PredictorBackend
import s3fs
import os

import av
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

fs = s3fs.S3FileSystem(anon=False)
AWS_LAMBDA_URL = os.getenv("AWS_LAMBDA_URL")

# Demo constants
DEMO_VIDEO_URL = "https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4"
DEMO_VIDEO_LABEL = "before"

# S3 constants
S3_BUCKET_NAME = "sign-recognizer"
S3_UPLOADED_VIDEOS_FOLDER = "new-videos"

st.header("Welcome to Gesto AI")
st.write("Upload any video of a sign or enter a public video URL and get a predicted word as text! The demo video for this app is [05727.mp4](https://sign-recognizer.s3.amazonaws.com/new-videos/05727.mp4) (prediction = 'before') from the WLASL dataset.")

input_video_url = st.text_input('Please enter a public URL pointing directly to a video:')
if st.button('Click here for sample video URLs'):
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05742.mp4", language="html")
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05740.mp4", language="html")
    st.code("https://sign-recognizer.s3.amazonaws.com/new-videos/05732.mp4", language="html")

video_url = None

# Option 1: Video file upload
uploaded_video = st.file_uploader("Or upload an .mp4 video file:")

if uploaded_video is not None:
    # We'll need this path for opening the video with OpenCV
    short_s3_video_url = f"{S3_BUCKET_NAME}/{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"    
    video_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{S3_UPLOADED_VIDEOS_FOLDER}/{uploaded_video.name}"
    print(f"Video S3 URL: {video_url}")

    # Save video to our S3 bucket
    with fs.open(short_s3_video_url, mode='wb') as f:
        f.write(uploaded_video.read()) 

    # Open video from disk path - technically not needed because we can feed the bytes-like object to st.video
    st.write(f"Uploaded video to AWS S3!")

# Option 2: Video URL 
elif input_video_url:
    print(f"Public URL for video: {input_video_url}")
    video_url = input_video_url

if video_url is not None:
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

st.write("Or use your webcam")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        flipped = img[:,::-1,:] 

        return av.VideoFrame.from_ndarray(flipped, format="bgr24")

def out_recorder_factory() -> MediaRecorder:
    return MediaRecorder("user_recording.mp4", format="mp4")

def stop_button():
    print("user webcam recording stopped!")
    # video_file = open('user_recording.mp4', 'rb')
    # video_bytes = video_file.read()

    # st.video(video_bytes)


webrtc_streamer(
    key="loopback",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_processor_factory=VideoProcessor,
    out_recorder_factory=out_recorder_factory,
    on_video_ended=stop_button
)