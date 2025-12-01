import os.path
import pickle
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, List

from torch import Tensor
from torchvision.datasets import Kinetics
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips


class StoreMetadataVideoClips(VideoClips):
    def __init__(self,
                 metadata_path: str,
                 video_paths: List[str],
                 clip_length_in_frames: int,
                 frames_between_clips: int,
                 frame_rate: Optional[int] = None,
                 _precomputed_metadata: Optional[Dict[str, Any]] = None,
                 num_workers: int = 0,
                 output_format: str = "THWC"):
        self.metadata_path: str = metadata_path
        super().__init__(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=_precomputed_metadata,
            num_workers=num_workers,
            output_format=output_format)

    def _compute_frame_pts(self) -> None:
        super()._compute_frame_pts()
        metadata = self.metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)


class KineticsDs(Kinetics):

    def __init__(self,
                 root: str,
                 frames_per_clip: int,
                 video_clips_pkl_name: str,
                 split: str,
                 frame_rate: Optional[int] = None,
                 step_between_clips: int = 1,
                 transform: Optional[Callable] = None,
                 label_transform: Optional[Callable] = None,
                 ):
        video_clips_pkl_path = os.path.join(root, 'video_clips_cache', video_clips_pkl_name)
        with open(video_clips_pkl_path, 'rb') as file:
            _precomputed_metadata: Optional[Dict[str, Any]] = pickle.load(file)
        super().__init__(root=root,
                         frames_per_clip=frames_per_clip,
                         num_classes='600',
                         split=split,
                         frame_rate=frame_rate,
                         step_between_clips=step_between_clips,
                         transform=transform,
                         _precomputed_metadata=_precomputed_metadata)
        self.label_transform: Optional[Callable] = label_transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, _, __, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return video, label

    @classmethod
    def calculate_metadata(cls,
                           pkl_name: str,
                           root: str,
                           frames_per_clip: int,
                           split: str,
                           step_between_clips: int,
                           extensions: Tuple[str, ...] = ("avi", "mp4"),
                           frame_rate: Optional[int] = 30,
                           num_workers: int = 1,
                           output_format: str = "TCHW",
                           ) -> None:
        split_folder = path.join(root, split)
        classes, class_to_idx = find_classes(split_folder)
        samples = make_dataset(split_folder, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in samples]
        StoreMetadataVideoClips(
            metadata_path=os.path.join(root, 'video_clips_cache', pkl_name),
            video_paths=video_list,
            clip_length_in_frames=frames_per_clip,
            frames_between_clips=step_between_clips,
            frame_rate=frame_rate,
            num_workers=num_workers,
            output_format=output_format
        )