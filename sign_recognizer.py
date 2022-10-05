"""Detects a word from the provided asl video.
Example usage as a script:
  python model_serve/sign_recognizer.py \
    /path/to/video.mp4
  python text_recognizer/paragraph_text_recognizer.py \
    https://fsdl-public-assets.s3-us-west-2.amazonaws.com/path/to/video.mp4
"""
import argparse
from pathlib import Path
from typing import Sequence, Union

import torch
from PIL import Image
from torchvision import transforms

from data_processing.wlasl_videos import *
from model.inception3d import *

# STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "paragraph-text-recognizer"
# MODEL_FILE = "model.pt"

LABEL_MAPPING_PATH = "./data_processing/wlasl_class_list.txt"
ID3_PRETRAINED_WEIGHTS_PATH = "./models/WLASL/weights/rgb_imagenet.pt"
WLASL_PRETRAINED_WEIGHTS_PATH = "./models/WLASL/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
NUM_CLASSES = 100


class ASLWordRecognizer:
    """Recognizes a word from sign in a video."""

    def __init__(self, model_path=None, mapping_path=None):
        # if model_path is None:
        #     model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        # self.model = torch.jit.load(model_path)
        if mapping_path is None:
            mapping_path = LABEL_MAPPING_PATH

        print("loading model")
        # self.model = load_inception_model()
        self.model = torch.jit.load(model_path)

        print("loading mapping")
        self.mapping = load_mapping(mapping_path)

    @torch.no_grad()
    def predict(self, video_filepath: Union[str, Path]) -> str:
        """Predict/infer word in input video (which can be a file path or url)."""
        print("processing video")
        batched_frames = process_video(video_filepath, 1, 74)

        print("generating predictions")
        y_pred = self.predict_on_video(batched_frames)

        pred_str = convert_y_label_to_string(y=y_pred[-1], mapping=self.mapping)

        return pred_str

    @torch.no_grad()
    def predict_on_video(self, batched_frames):
        """
        Args:
            model: InceptionI3d
            batched_frames: torch.Tensor: [batch_num, num_channels, num_frames, height, width]
        Returns:
            top_prediction: int
        """
        # Inference time! Get the logits for the single video example
        per_frame_logits = self.model(batched_frames)

        # Batch size, number of classes, unclear what the 3rd number/dimension is here?
        # st.write(f"Logits shape: {per_frame_logits.shape}")

        # Predictions from the logits
        predictions = torch.max(per_frame_logits, dim=2)[0]
        # st.write(f"Predictions shape after getting the max per the third dimension: {predictions.shape}")

        # Sort predictions in a descending order of relevance - the top prediction is at the end
        # This is an array of indices!
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        # st.write(f"Output predictions: {out_labels}")

        # # Return the top prediction
        # return out_labels[-1]

        # Return all predictions
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
    # return "".join([mapping[int(i)] for i in y])
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
    # st.write(f"Input frames: {type(frames)}, {frames.shape}")

    # Center video on hand
    img_transform = transforms.Compose([MyCenterCrop(224)])
    transformed_frames = img_transform(frames)
    # st.write(f"Transformed frames: {type(transformed_frames)}, {transformed_frames.shape}")

    # Convert to Torch Tensor and reshape
    tensor_frames = video_to_tensor(transformed_frames)
    # st.write(f"Tensor frames: {type(tensor_frames)}, {tensor_frames.shape}")

    # Add an extra dimension for making this video a part of a batch
    batched_frames = torch.from_numpy(np.expand_dims(tensor_frames, axis=0))
    # st.write(f"Batched frames: {type(batched_frames)}, {batched_frames.shape}")

    return batched_frames


def load_inception_model(device=0):
    """
    Args:
        device: int
    Returns:
        pretrained_i3d_model: InceptionI3d
    """

    # Initialize model
    pretrained_i3d_model = torch.jit.script(InceptionI3d(100, in_channels=3))

    # # Load the general inception model weights
    # pretrained_i3d_model.load_state_dict(
    #     torch.load(ID3_PRETRAINED_WEIGHTS_PATH), strict=False
    # )

    # Adapt the final layer for the number of classes we expect
    # pretrained_i3d_model.replace_logits(NUM_CLASSES)

    # Load the weights for the fine-tuned model on ASL
    pretrained_i3d_model.load_state_dict(
        torch.load(WLASL_PRETRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")),
        strict=False,
    )

    # Move to GPU
    # i3d.cuda(device=device)

    # Add data parallelism layer (but this is not actually that useful here since we're only using 1 example)
    # pretrained_i3d_model = nn.DataParallel(pretrained_i3d_model)

    # Put model in inference mode
    pretrained_i3d_model.eval()

    # pretrained_model = PL_resnet50.load_from_checkpoint(checkpoint_path=weight_loc).eval()#.cuda(device=0)
    return pretrained_i3d_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "model_path", type=str, help="location of pytorch torchscript model"
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Name for a video file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    sign_recognizer = ASLWordRecognizer(args.model_path)
    pred_str = sign_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
