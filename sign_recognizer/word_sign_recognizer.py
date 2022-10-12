"""Detects a word from the provided ASL video.
Example usage as a script (currently NOT working due to relative import path errors):
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

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "sign-recognizer"
MODEL_FILE = "model.pt"
FULL_MODEL_PATH = STAGED_MODEL_DIRNAME / MODEL_FILE

if not path.exists(FULL_MODEL_PATH):
    raise FileNotFoundError(f"Torchscript model file not found! Expected: {FULL_MODEL_PATH}")

LABEL_MAPPING_PATH = (
    Path(__file__).resolve().parent / "data_processing" / "wlasl_class_list.txt"
)


class ASLWordRecognizer:
    """Recognizes a word from sign in a video."""

    def __init__(self, model_path=None, mapping_path=None):
        if model_path is None:
            model_path = FULL_MODEL_PATH

            print(f"Found torchscript model path: {model_path}")

        print("Loading model...")
        self.model = torch.jit.load(model_path)

        if mapping_path is None:
            mapping_path = LABEL_MAPPING_PATH

        print("Loading label mapping...")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # parser.add_argument(
    #     "model_path",
    #     type=str,
    #     help="location of pytorch torchscript model",
    # )
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
