import numbers
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import torch


def video_to_tensor(pic: np.ndarray) -> torch.Tensor:
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames_from_video_dataset(video_path: Union[str, Path], start_frame: int = 0, num_frames: int = -1) -> np.ndarray:

    vidcap = cv2.VideoCapture(video_path)

    frames = []
    if num_frames == -1:
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"INFO Number of frames on video: {num_frames}")
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames):
        _, img = vidcap.read()

        if img is not None:
            w, h, _ = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.) * 2 - 1

            frames.append(img)

    return np.asarray(frames, dtype=np.float32)


class MyCenterCrop(object):
    """Crops the given seq Images at the center.
    """

    def __init__(self, size: Union[int, Sequence]):
        """
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Crop input video frames
        Args:
            frames: np.array
                Video frames to be cropped.
        Returns:
            frames: np.array
                Cropped video frames.
        """
        t, h, w, c = img.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return img[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
