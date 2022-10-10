import streamlit as st

from sign_recognizer.word_sign_recognizer import ASLWordRecognizer

st.header("Welcome to Gesto AI")
st.write("Upload any video of a sign and get a predicted word as text! The demo video for this app is 05727.mp4 from the WLASL dataset.")

uploaded_video = st.file_uploader("Upload a video...")

if uploaded_video is not None:
    pretrained_model = ASLWordRecognizer()

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
    st.write("Uploaded video and stored to disk! Getting predictions...")

    # Temporarily hard coded for the video we're uploading/testing
    label = 3

    res = pretrained_model.predict(video_filepath)
    st.write(f"Final prediction: {res}")
    st.write(f"Expected label: {pretrained_model.mapping[label]}")
    
    
    correctness_state = st.selectbox('Would you like to submit feedback?',
                ('Predicted word is correct!', 'Predicted word is incorrect.'))    
    if correctness_state == 'Predicted word is incorrect.':
        st.write("Please tell us what the correct word was. Check back soon for an updated model that learns from your feedback")
        correct_label = st.text_input('Enter word here', "Hello")
        st.write('The current correct_label', correct_label)


    
