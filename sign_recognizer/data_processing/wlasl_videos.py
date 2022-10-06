import json
import os
import os.path

import numbers
import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames_from_video_dataset(video_path, start=0, num=-1):
    # video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')

    frames = []
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(num):
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

    
class SingleVideoNSLTDataset(data_utl.Dataset):

    def __init__(self, video_tensor, label, start_frame, num_frames, video_name, transforms=None):
        self.data = [(video_tensor, label, start_frame, num_frames, video_name)]
        self.transforms = transforms

    def __getitem__(self, index):
        vid, label, start_f, start_e, output_name = self.data[index]
        return vid, label, vid

    def __len__(self):
        return len(self.data)

class MyCenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)