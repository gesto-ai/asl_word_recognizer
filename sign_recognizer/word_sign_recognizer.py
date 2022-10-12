"""Detects a word from the provided ASL video.
Example usage as a script:
  python sign_recognizer/word_sign_recognizer.py \
    /path/to/video.mp4
  python sign_recognizer/word_sign_recognizer.py \
    https://fsdl-public-assets.s3-us-west-2.amazonaws.com/path/to/video.mp4
"""
import argparse
from os import path
from pathlib import Path
from typing import Sequence, Union

import torch
from torchvision import transforms

from sign_recognizer.data_processing.wlasl_videos import *
from sign_recognizer.model.inception3d import *

# This defines the parent folder of this file, and adds "/artifacts" to the path string
# An example would be BASE_DIRNAME = `/Users/dafirebanks/GestoAI/model_serve/sign_recognizer/`
BASE_DIRNAME = Path(__file__).resolve().parent
ARTIFACTS_DIRNAME = BASE_DIRNAME / "artifacts" 

# Then here we only define the subdirectories/filenames of the objects in the sign_recognizer/artifacts folder
ID3_PRETRAINED_WEIGHTS_PATH = "models/WLASL/weights/rgb_imagenet.pt"
WLASL_PRETRAINED_WEIGHTS_PATH = "models/WLASL/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"

# Make sure model weights paths exist
if not path.exists(ARTIFACTS_DIRNAME / ID3_PRETRAINED_WEIGHTS_PATH):
    raise FileNotFoundError(f"Inception I3D model weights file not found! Expecting: {ARTIFACTS_DIRNAME / ID3_PRETRAINED_WEIGHTS_PATH}")

if not path.exists(ARTIFACTS_DIRNAME / WLASL_PRETRAINED_WEIGHTS_PATH):
    raise FileNotFoundError(f"WLASL model weights file not found! Expecting: {ARTIFACTS_DIRNAME / WLASL_PRETRAINED_WEIGHTS_PATH}")

# Mapping file from label numbers to actual text
LABEL_MAPPING_PATH = BASE_DIRNAME / "data_processing" / "wlasl_class_list.txt"
NUM_CLASSES = 100


class ASLWordRecognizer:
    """Recognizes a word from sign in a video."""

    def __init__(self, id3_model_path=None, wlasl_model_path=None, mapping_path=None, num_classes=None):
        if mapping_path is None:
            mapping_path = LABEL_MAPPING_PATH
        if id3_model_path is None:
            id3_model_path = ARTIFACTS_DIRNAME / ID3_PRETRAINED_WEIGHTS_PATH
        if wlasl_model_path is None:
            wlasl_model_path = ARTIFACTS_DIRNAME / WLASL_PRETRAINED_WEIGHTS_PATH
        if num_classes is None:
            num_classes = NUM_CLASSES
        
        print("Loading model...")
        self.model = load_inception_model(id3_model_path, wlasl_model_path, num_classes)

        print("Loading mapping...")
        self.mapping = load_mapping(mapping_path)

    @torch.no_grad()
    def predict(self, video_filepath: Union[str, Path]) -> str:
        """Predict/infer word in input video (which can be a file path or url)."""
        print("Processing video...")
        batched_frames = process_video(video_filepath, 1, 74)

        print("Generating predictions...")
        y_pred = self.predict_on_video(batched_frames)
        pred_str = convert_y_label_to_string(y=y_pred[-1], mapping=self.mapping)

        return pred_str
    
    @torch.no_grad()
    def predict_on_video(self, batched_frames):
        """Predict word in video passed as a tensor of frames
        Args:
            model: InceptionI3d
            batched_frames: torch.Tensor: [batch_num, num_channels, num_frames, height, width]
        Returns:
            top_prediction: int
        """

        # Inference time! Get the logits for the single video example
        print("-> Step 1: Getting the logits")
        per_frame_logits = self.model(batched_frames)

        # Predictions from the logits
        print("-> Step 2: Get the predictions from the logits")
        predictions = torch.max(per_frame_logits, dim=2)[0]

        # Sort predictions in a descending order of relevance - the top prediction is at the end
        # This is an array of indices!
        print("-> Step 3: Sort predictions from least relevant to most relevant")
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

        # Return the top prediction
        return out_labels

def load_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, "r") as f:
        labels: list = f.readlines()
    
    for label in labels:
        idx, ann = label.split("\t")
        mapping[int(idx)] = ann.replace("\n", "")
    
    return mapping

def convert_y_label_to_string(y: np.int64, mapping: Sequence[str]) -> str:
    return mapping[y]

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
    # print(f"Input frames: {type(frames)}, {frames.shape}")

    # Center video on hand
    img_transform = transforms.Compose([MyCenterCrop(224)])
    transformed_frames = img_transform(frames)
    # print(f"Transformed frames: {type(transformed_frames)}, {transformed_frames.shape}")

    # Convert to Torch Tensor and reshape
    tensor_frames = video_to_tensor(transformed_frames)
    # print(f"Tensor frames: {type(tensor_frames)}, {tensor_frames.shape}")

    # Add an extra dimension for making this video a part of a batch
    batched_frames = torch.from_numpy(np.expand_dims(tensor_frames, axis=0))
    # print(f"Batched frames: {type(batched_frames)}, {batched_frames.shape}")

    return batched_frames



def load_inception_model(id3_pretrained_weights_path, wlasl_pretrained_weights_path, num_classes, device=0):
    """
    Args:
        device: int
    Returns:
        pretrained_i3d_model: InceptionI3d
    """
    
    # Initialize model
    pretrained_i3d_model = InceptionI3d(400, in_channels=3)

    # Load the general inception model weights
    pretrained_i3d_model.load_state_dict(torch.load(id3_pretrained_weights_path))

    # Adapt the final layer for the number of classes we expect
    pretrained_i3d_model.replace_logits(num_classes)

    # Load the weights for the fine-tuned model on ASL
    pretrained_i3d_model.load_state_dict(torch.load(wlasl_pretrained_weights_path, map_location=torch.device('cpu')))

    # Move to GPU
    # i3d.cuda(device=device)

    # Add data parallelism layer (but this is not actually that useful here since we're only using 1 example)
    pretrained_i3d_model = nn.DataParallel(pretrained_i3d_model)

    # Put model in inference mode
    pretrained_i3d_model.eval()

    # pretrained_model = PL_resnet50.load_from_checkpoint(checkpoint_path=weight_loc).eval()#.cuda(device=0)
    return pretrained_i3d_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for a video file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    sign_recognizer = ASLWordRecognizer()
    pred_str = sign_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
