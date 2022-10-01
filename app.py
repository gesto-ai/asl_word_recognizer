import streamlit as st
import numpy as np
import sys
from torchvision import transforms

sys.path.append("../")

# from asl_sign_spelling_classification.train_v0 import * 
from model.inception3d import *
from data_processing.wlasl_videos import *

st.header("Welcome to Gesto AI")
st.write("Upload any video of a sign and get a predicted word as text! The demo video for this app is 05727.mp4 from the WLASL dataset.")

uploaded_video = st.file_uploader("Upload a video...")

 # We'll use the weights from the general pre-trained inception model, then the weights for the fine tuned one on ASL
 # Download these from https://github.com/dxli94/WLASL#training-and-testing
ID3_PRETRAINED_WEIGHTS_PATH = "GestoAI/models/WLASL/weights/rgb_imagenet.pt"
WLASL_PRETRAINED_WEIGHTS_PATH = "GestoAI/models/WLASL/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
NUM_CLASSES = 100

def load_inception_model(device=0):
    """
    Args:
        device: int
    Returns:
        pretrained_i3d_model: InceptionI3d
    """
    
    # Initialize model
    pretrained_i3d_model = InceptionI3d(400, in_channels=3)

    # Load the general inception model weights
    pretrained_i3d_model.load_state_dict(torch.load(ID3_PRETRAINED_WEIGHTS_PATH))

    # Adapt the final layer for the number of classes we expect
    pretrained_i3d_model.replace_logits(NUM_CLASSES)

    # Load the weights for the fine-tuned model on ASL
    pretrained_i3d_model.load_state_dict(torch.load(WLASL_PRETRAINED_WEIGHTS_PATH, map_location=torch.device('cpu')))

    # Move to GPU
    # i3d.cuda(device=device)

    # Add data parallelism layer (but this is not actually that useful here since we're only using 1 example)
    pretrained_i3d_model = nn.DataParallel(pretrained_i3d_model)

    # Put model in inference mode
    pretrained_i3d_model.eval()

    # pretrained_model = PL_resnet50.load_from_checkpoint(checkpoint_path=weight_loc).eval()#.cuda(device=0)
    return pretrained_i3d_model

def process_video(video_filepath, start_frame, end_frame):
    """
    Args:
        video_filepath: str
        start_frame: int
        end_frame: int
    Returns:
        batched_frames: torch.Tensor: [batch_num, num_channels, num_frames, height, width]
    """
    # Load frames
    frames = load_rgb_frames_from_video_dataset(video_filepath, start_frame, end_frame)
    st.write(f"Input frames: {type(frames)}, {frames.shape}")

    # Center video on hand
    img_transform = transforms.Compose([MyCenterCrop(224)])
    transformed_frames = img_transform(frames)
    st.write(f"Transformed frames: {type(transformed_frames)}, {transformed_frames.shape}")

    # Convert to Torch Tensor and reshape
    tensor_frames = video_to_tensor(transformed_frames)
    st.write(f"Tensor frames: {type(tensor_frames)}, {tensor_frames.shape}")

    # Add an extra dimension for making this video a part of a batch
    batched_frames = torch.from_numpy(np.expand_dims(tensor_frames, axis=0))
    st.write(f"Batched frames: {type(batched_frames)}, {batched_frames.shape}")

    return batched_frames

def predict_on_video(model, batched_frames):
    """
    Args:
        model: InceptionI3d
        batched_frames: torch.Tensor: [batch_num, num_channels, num_frames, height, width]
    Returns:
        top_prediction: int
    """
    # Inference time! Get the logits for the single video example
    per_frame_logits = model(batched_frames)

    # Batch size, number of classes, unclear what the 3rd number/dimension is here?
    st.write(f"Logits shape: {per_frame_logits.shape}")

    # Predictions from the logits
    predictions = torch.max(per_frame_logits, dim=2)[0]
    st.write(f"Predictions shape after getting the max per the third dimension: {predictions.shape}")

    # Sort predictions in a descending order of relevance - the top prediction is at the end
    # This is an array of indices!
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    st.write(f"Output predictions: {out_labels}")

    # Return the top prediction
    return out_labels[-1]

if uploaded_video is not None:
    pretrained_model = load_inception_model()

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

    # Temporarily hard coded for the video we're uploading/testing
    label = 3
    start_frame = 1
    end_frame = 74

    # Create a batch of frames from video
    batched_frames = process_video(video_filepath, start_frame, end_frame)

    # Make a prediction on video
    res = predict_on_video(pretrained_model, batched_frames)
    st.write(f"Final prediction: {res}")
    st.write(f"Expected label: {label}")


    
