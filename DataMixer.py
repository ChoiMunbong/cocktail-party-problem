import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class SourceMixer(object):
	def __init__(self, n_src, samples, frames, seed=42):
		self.n_src = n_src
		self.samples = samples
		self.frames = frames
		self.seed = seed

	def mix(self, x, y, shift_factor=None, shift_overlaps=False, **kwargs):
		random.seed(self.seed)
		np.random.seed(self.seed)
  
		ids, counts = np.unique(y, return_counts=True) # 각각의 데이터에 대해 label이 몇인지 / 몇개가 있는지 (index를 key(label)로봄)
		ids, counts = ids.tolist(), counts.tolist() #list로 변경

		x_mix = []
		y_mix = []
		y_mix_id = []
		
		xy_dict = {i: [x_i for (x_i,y_i) in zip(x, y) if y_i==i] for i in ids} #클래스에 대해서 i = key(label) 나머지 value(list) 

		for n in tqdm.tqdm(range(self.samples)):
			id_idxs = random.sample(ids, self.n_src) #입력 데이터 랜덤 추출
			# id_idxs = ids
			signal_idxs = [np.random.randint(low=0, 
											 high=counts[ids.index(id_idx)]) for id_idx in id_idxs] #각 id_idx에 대해서, 그 id_idx에 해당하는 클래스의 count 수 내에서 랜덤한 정수를 선택합니다. 이것은 랜덤하게 선택한 클래스에서 랜덤하게 데이터를 선택하기 위한 인덱스입니다.

			signals = [xy_dict[id_idx][signal_idx] for (id_idx, signal_idx) in zip(id_idxs, signal_idxs)] #위에서 선택한 id_idx와 signal_idx를 이용해 각 클래스에서 데이터를 선택합니다.

			if shift_factor is not None:
				signals = [self.signal_shifter(signal,
											   shift_factor=shift_factor,
											   pad=kwargs['pad'],
											   side=kwargs['side']) for signal in signals]

			if shift_overlaps:
				assert kwargs['pad'] == 'zero', print('Overlap shift is compatible with zero pad only')

				signals = self.overlap_shifter(signals)

			signals = [torch.Tensor(s) for s in signals] 
			x_mix.append(sum(signals).view(1, -1)) # 두 signal 결합
			y_mix.append(torch.stack(signals, dim=0))
			y_mix_id.append(torch.LongTensor(id_idxs))

		return x_mix, y_mix, y_mix_id
	
	@staticmethod
	def signal_shifter(signal, shift_factor, pad='zero', side='front'):
		if type(pad)==str:
			assert pad in ['zero', 'noise', 'sub_noise'], print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
		else:
			assert type(pad) == float, print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
		assert side in ['front', 'back']

		frames = signal.shape[-1]

		if pad == 'zero':
			noise = 0
		elif pad == 'noise':
			noise = signal[-1]
		elif pad == 'sub_noise':
			noise = signal
			signal = signal - noise
			noise = 0
		else:
			noise = pad

		frame_shift = random.randint(0, int(shift_factor * frames))
		noise_frames = np.ones(frame_shift, dtype='float32') * noise

		if side == 'front':
			signal_frames = signal[0:frames - frame_shift]
		elif side == 'back':
			signal_frames = signal[frame_shift-1:-1]

		signal = np.concatenate([noise_frames, signal_frames]).astype('float32')

		return signal

	@staticmethod
	def overlap_shifter(signals):
		frames = signals[0].shape[-1]

		back_trimmed = [np.trim_zeros(s, 'b') for s in signals]

		back_trimmed_lengths = [len(b) for b in back_trimmed]
		max_length = max(back_trimmed_lengths)
		max_index = back_trimmed_lengths.index(max_length)

		front_zeros = [frames - len(np.trim_zeros(s, 'f')) for s in signals]

		for i, s in enumerate(signals):
			if i != max_index:
				max_shift = min(max_length - front_zeros[i], frames -back_trimmed_lengths[i])
				min_shift = max(0, front_zeros[max_index] - back_trimmed_lengths[i])
				shift = np.zeros(np.random.randint(low=min_shift, high=np.max([max_shift, min_shift+1])), dtype='float32')
				signals[i] = np.concatenate([shift, s]).astype('float32')[:frames]
		return signals