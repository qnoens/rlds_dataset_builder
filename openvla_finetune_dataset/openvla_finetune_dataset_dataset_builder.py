from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Added to read parquet files
import pandas as pd
import cv2


class OpenvlaFinetuneDataset(tfds.core.GeneratorBasedBuilder):
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
                    'tcp_pose': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='EEF pose of robot, same as state (added)',
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
            'train': self._generate_examples(path='/fast_storage/qnoens/OpenVLA/data/train_push_button_easy/episode_*.parquet'),
            'val': self._generate_examples(path='/fast_storage/qnoens/OpenVLA/data/val_push_button_easy/episode_*.parquet'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            data = pd.read_parquet(episode_path, engine='pyarrow') # this is a pandas dataframe containing all the features

            episode_index = None
            task_index = None

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            data_len = data.shape[0]
            for i in range(1, data_len):
                step = data.loc[i]
                # compute Kona language embedding
                language_embedding = self._embed(['push the button'])[0].numpy() # not sure if this is correct; just replaced the string by step['language_instruction']. OpenVLA does not use this anyway, it uses its own language embedding

                # load image
                image_dict = step['observation.images.Camera_rgb_image']
                image_np = np.frombuffer(image_dict['bytes'], dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                # load wrist image
                wrist_image_dict = step['observation.images.ur5e_WristCamera_rgb_image']
                wrist_image_np = np.frombuffer(wrist_image_dict['bytes'], dtype=np.uint8)
                wrist_image = cv2.imdecode(wrist_image_np, cv2.IMREAD_COLOR)

                episode.append({
                    'observation': {
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': step['observation.state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': step['next.reward'],
                    'seed': step['seed'],
                    'timestamp': step['timestamp'],
                    'tcp_pose': step['ur5e/tcp_pose'],
                    'frame_index': step['frame_index'],
                    'success': step['next.success'],
                    'index': step['index'],
                    'is_first': step['frame_index'] == 0,
                    'is_last': i == (data_len - 1), 
                    'is_terminal': i == (data_len - 1),
                    'language_instruction': 'push the button', #step['language_instruction']
                    'language_embedding': language_embedding,
                })

                episode_index = step['episode_index']
                task_index = step['task_index']

            # This code was added because some episodes does not seem to be useful in the dataset. This code catches these.
            try:
                test = data.loc[1]['episode_index']
            except:
                print(f"Episode {episode_path} is not valid. Skipping this episode.")
                return None

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_index': data.loc[1]['episode_index'], #Raised error, so temporarily removed. Not a problem as long as we stick with one task
                    'task_index': data.loc[1]['task_index'],
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

