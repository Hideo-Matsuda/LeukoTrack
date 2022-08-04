import os.path as osp

import torch
from torch.utils.data import Dataset

from .mot_sequence import MOTSequence


class MOT17Wrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dets, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		mot_dir = 'MOT17'
		train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
		test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT17-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT17-{split}"]
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			if dets == 'ALL':
				self._data.append(MOTSequence(f"{s}-DPM", mot_dir, **dataloader))
				self._data.append(MOTSequence(f"{s}-FRCNN", mot_dir, **dataloader))
				self._data.append(MOTSequence(f"{s}-SDP", mot_dir, **dataloader))
			elif dets == 'DPM16':
				self._data.append(MOTSequence(s.replace('17', '16'), 'MOT16', **dataloader))
			else:
				self._data.append(MOTSequence(f"{s}-{dets}", mot_dir, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOT19Wrapper(MOT17Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['MOT19-01', 'MOT19-02', 'MOT19-03', 'MOT19-05']
		test_sequences = ['MOT19-04', 'MOT19-06', 'MOT19-07', 'MOT19-08']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT19-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT19-{split}"]
		else:
			raise NotImplementedError("MOT19CVPR split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOTSequence(s, 'MOT19', **dataloader))


class MOT20Wrapper(MOT17Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
		#train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05','MOT20-01_1', 'MOT20-02_1', 'MOT20-03_1', 'MOT20-05_1', 'MOT20-01_2', 'MOT20-02_2', 'MOT20-03_2', 'MOT20-05_2', 'MOT20-01_3', 'MOT20-02_3', 'MOT20-03_3', 'MOT20-05_3']
		#test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
		test_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT20-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT20-{split}"]
		else:
			raise NotImplementedError("MOT20 split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOTSequence(s, 'MOT20', **dataloader))


class MOT17LOWFPSWrapper(MOT17Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""

		sequences = ['MOT17-02', 'MOT17-04', 'MOT17-09', 'MOT17-10', 'MOT17-11']

		self._data = []
		for s in sequences:
			self._data.append(
				MOTSequence(f"{s}-FRCNN", osp.join('MOT17_LOW_FPS', f'MOT17_{split}_FPS'), **dataloader))


class MOT17PrivateWrapper(MOT17Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader, data_dir):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
		test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT17-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT17-{split}"]
		else:
			raise NotImplementedError("MOT17 split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOTSequence(s, data_dir, **dataloader))

class BiodataWrapper(MOT17Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		mot_dir = 'biodata'
		G1_test_sequences = ['sample01', 'sample02', 'sample03'] # Sequence of test data names for group1.
		G2_test_sequences = ['sample04', 'sample05', 'sample06'] # Sequence of test data names for group2.
		all_sequences = ['sample01', 'sample02', 'sample03','sample04', 'sample05', 'sample06'] # Sequence of all data names including train data.
		sample_sequences = ['sample'] # seq of sample data name

		if "all" == split:
			sequences = all_sequences
		elif "group1" == split: 
			sequences = G1_test_sequences
		elif "group2" == split: 
			sequences = G2_test_sequences
		# Edit code by imitating "sample". (if you want to suit for original evaluation method)
		elif "sample" == split:
			sequences = sample_sequences
		else:
			raise NotImplementedError("biodata split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOTSequence(s, mot_dir, **dataloader))