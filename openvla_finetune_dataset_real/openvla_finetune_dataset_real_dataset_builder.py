from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import logging

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print("imports done.")

root_dir = '/fast_storage/qnoens/OpenVLA/data/lerobot_blue_only_zero_actions2'
dataset_name = 'test_dataset'

dataset = LeRobotDataset(repo_id=dataset_name, root=root_dir)

check = dataset[0]["episode_index"] == 0
print("Dataset loaded", check)


# Added to read parquet files
import pandas as pd
import cv2


class OpenvlaFinetuneDatasetReal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({ # ['next.reward', 'next.success', 'seed', 'timestamp', 'observation.images.Camera_rgb_image', 'observation.images.ur5e_WristCamera_rgb_image','ur5e/joint_configuration', 'observation.state', 'action', 'frame_index', 'episode_index', 'index', 'task_index']
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of EEF XYZ (3) + Quaternion (4)',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1) ',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                    'seed': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Seed used (added)'
                    ),
                    'timestamp': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Seed used (added)'
                    ),
                    'frame_index': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Frame index (added)',
                    ),
                    'success': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if episode was successful (added)',
                    ),
                    'index': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Index of current step (added)',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_index': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Episode index (added)',
                    ),
                    'task_index': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Task index (added)',
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/fast_storage/qnoens/OpenVLA/data/lerobot_blue_only_zero_actions2/data/chunk-000/episode_*.parquet'),
            #'val': self._generate_examples(path='/fast_storage/qnoens/OpenVLA/data/full_lerobot_task1/data/chunk-000/episode_*.parquet'),
        }
    
    # New one
    def _generate_examples(self, path):# -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, dataset):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            data = pd.read_parquet(episode_path, engine='pyarrow') # this is a pandas dataframe containing all the features
            ep_idx = int(episode_path.split("/")[-1].split("_")[1].split(".")[0])
            logging.info(f"Parsing episode {ep_idx} from {episode_path}")

            # Last workaround I can think of: take the orginal datasets whithout the deletes, this should cause no issues. Fix issue during training...
            # if ep_idx in [31, 32, 33, 34, 35, 36, 37, 38, 65, 74, 91]:
            #     logging.info(f"Skipping episode {ep_idx} because it is deleted.")
            #     return None, None


            meta = dataset.meta
            number_of_episodes = len(meta.episodes)

            #episode_data = meta.episodes_stats[ep_idx]

            lengths = [meta.episodes[i]['length'] for i in range(ep_idx)]
            start_idx = sum(lengths)
            # start_idx = episode_data['index']['min']
            # end_idx = episode_data['index']['max']

            frame_nr = start_idx
            logging.info(f"Frame number: {frame_nr} of type {type(frame_nr)}")
            episode = []
            for i in range(meta.episodes[ep_idx]["length"]):
                frame = dataset[frame_nr]

                # All the data conversion happens here
                # Scene image
                scene_image = frame["scene_image"]
                scene_image = scene_image.permute(1, 2, 0).numpy()
                scene_image = (scene_image * 255).astype(np.uint8)

                # Wrist image
                wrist_image = frame["wrist_image"]
                wrist_image = wrist_image.permute(1, 2, 0).numpy()
                wrist_image = (wrist_image * 255).astype(np.uint8)

                language_instruction = meta.episodes[ep_idx]["tasks"][0]
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'image': scene_image,
                        'wrist_image': wrist_image,
                        'state': frame['state'],
                    },
                    'action': frame['action'],
                    'discount': 1.0,
                    'reward': frame['next.reward'],
                    'seed': frame['seed'],
                    'timestamp': frame['timestamp'],
                    'frame_index': frame['frame_index'],
                    'success': frame['next.success'],
                    'index': frame['index'],
                    'is_first': frame['frame_index'] == 0,
                    'is_last': i == (meta.episodes[ep_idx]["length"] - 1), 
                    'is_terminal': i == (meta.episodes[ep_idx]["length"] - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

                frame_nr += 1

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_paths[ep_idx],
                    'episode_index': ep_idx,
                    'task_index': meta.get_task_index(language_instruction),
                }
            }


            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)


        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            episode_path, sample = _parse_example(sample, dataset)
            if sample is None:
                logging.info(f"Skipping...")
                continue
            yield episode_path, sample

