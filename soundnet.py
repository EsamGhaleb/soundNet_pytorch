import numpy as np
import torch
import torch.nn as nn
from torch import bmm, cat, randn, zeros
from torch.autograd import Variable
import os
from util import load_from_txt

LEN_WAVEFORM = 22050 * 20

local_config = {
	'batch_size': 1,
	'eps': 1e-5,
	'sample_rate': 22050,
	'load_size': 22050 * 20,
	'name_scope': 'SoundNet_TF',
	'phase': 'extract',
}


class SoundNet(nn.Module):
	def __init__(self):
		super(SoundNet, self).__init__()
		
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0))
		print("Conv1", self.conv1.weight.shape, self.conv1.bias.shape)
		self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
		print("Bn1", self.batchnorm1.weight.shape, self.batchnorm1.bias.shape)
		self.relu1 = nn.ReLU(True)
		self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))
		
		self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0))
		print("Conv2", self.conv2.weight.shape, self.conv2.bias.shape)
		self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
		print("Bn2", self.batchnorm2.weight.shape, self.batchnorm2.bias.shape)
		self.relu2 = nn.ReLU(True)
		self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))
		
		self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0))
		print("Conv3", self.conv3.weight.shape, self.conv3.bias.shape)
		self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
		print("Bn3", self.batchnorm3.weight.shape, self.batchnorm3.bias.shape)
		self.relu3 = nn.ReLU(True)
		
		self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
		print("Conv4", self.conv4.weight.shape, self.conv4.bias.shape)
		self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
		print("Bn4", self.batchnorm4.weight.shape, self.batchnorm4.bias.shape)
		self.relu4 = nn.ReLU(True)
		
		self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
		print("Conv5", self.conv5.weight.shape, self.conv5.bias.shape)
		self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
		print("Bn5", self.batchnorm5.weight.shape, self.batchnorm5.bias.shape)
		self.relu5 = nn.ReLU(True)
		self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))
		
		self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
		print("Conv6", self.conv6.weight.shape, self.conv6.bias.shape)
		self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
		print("Bn6", self.batchnorm6.weight.shape, self.batchnorm6.bias.shape)
		self.relu6 = nn.ReLU(True)
		
		self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
		print("Conv7", self.conv7.weight.shape, self.conv7.bias.shape)
		self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
		print("Bn7", self.batchnorm7.weight.shape, self.batchnorm7.bias.shape)
		self.relu7 = nn.ReLU(True)
		
		self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))
		print("Conv81", self.conv8_objs.weight.shape, self.conv8_objs.bias.shape)
		self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1), stride=(2, 1))
		print("Conv82", self.conv8_scns.weight.shape, self.conv8_scns.bias.shape)
	
	def forward(self, waveform):
		"""
			Args:
				waveform (Variable): Raw 20s waveform.
		"""
		if torch.cuda.is_available():
			waveform.cuda()
		
		out = self.conv1(waveform)
		print('Max value of conv1: {:.4f}'.format(np.max(out.data.numpy())))
		print('Min value of conv1: {:.4f}'.format(np.min(out.data.numpy())))
		out = self.batchnorm1(out)
		print('Max value of BN1: {:.4f}'.format(np.max(out.data.numpy())))
		print('Min value of BN1: {:.4f}'.format(np.min(out.data.numpy())))
		out = self.relu1(out)
		print('Max value of relU1: {:.4f}'.format(np.max(out.data.numpy())))
		print('Min value of relu1: {:.4f}'.format(np.min(out.data.numpy())))
		out = self.maxpool1(out)
		print('Max value of maxpool1: {:.4f}'.format(np.max(out.data.numpy())))
		print('Min value of maxpool1: {:.4f}'.format(np.min(out.data.numpy())))
		
		return out.data.numpy()
	
	@staticmethod
def put_weights(batchnorm, conv, params_w, batch_norm=True):
	if batch_norm:
		bn_bs = params_w['beta']
		batchnorm.bias.data = torch.from_numpy(bn_bs)
		bn_ws = params_w['gamma']
		batchnorm.weight.data = torch.from_numpy(bn_ws)
		bn_mean = params_w['mean']
		batchnorm.running_mean.data = torch.from_numpy(bn_mean)
		bn_var = params_w['var']
		batchnorm.running_var.data = torch.from_numpy(bn_var)
	
	conv_bs = params_w['biases']
	conv.bias.data = torch.from_numpy(conv_bs)
	conv_ws = params_w['weights']
	conv.weight.data = torch.from_numpy(conv_ws).permute(3, 2, 0, 1)
	return batchnorm, conv
	
	def load_weights(self):
		param_G = np.load('models/sound8.npy', encoding='latin1').item()
		
		params_w = param_G['conv1']
		self.batchnorm1, self.conv1 = self.put_weights(self.batchnorm1, self.conv1, params_w)
		
		params_w = param_G['conv2']
		self.batchnorm2, self.conv2 = self.put_weights(self.batchnorm2, self.conv2, params_w)
		
		params_w = param_G['conv3']
		self.batchnorm3, self.conv3 = self.put_weights(self.batchnorm3, self.conv3, params_w)
		
		params_w = param_G['conv4']
		self.batchnorm4, self.conv4 = self.put_weights(self.batchnorm4, self.conv4, params_w)
		
		params_w = param_G['conv5']
		self.batchnorm5, self.conv5 = self.put_weights(self.batchnorm5, self.conv5, params_w)
		
		params_w = param_G['conv6']
		self.batchnorm6, self.conv6 = self.put_weights(self.batchnorm6, self.conv6, params_w)
		
		params_w = param_G['conv7']
		self.batchnorm7, self.conv7 = self.put_weights(self.batchnorm7, self.conv7, params_w)
		
		params_w = param_G['conv8']
		_, self.conv8_objs = self.put_weights([], self.conv8_objs, params_w, batch_norm=False)
		params_w = param_G['conv8_2']
		_, self.conv8_scns = self.put_weights([], self.conv8_scns, params_w, batch_norm=False)


# @mem.cache()
def extract_features():
	audio_txt = 'audio_files.txt'
	
	model = SoundNet()
	model.load_weights()
	
	# Extract Feature
	sound_samples, audio_paths = load_from_txt(audio_txt, config=local_config)
	
	print(LEN_WAVEFORM / 6)
	print(model)
	features = {}
	features['feats'] = []
	features['paths'] = []
	model.eval()
	for idx, sound_sample in enumerate(sound_samples):
		print(audio_paths[idx])
		new_sample = torch.from_numpy(sound_sample)
		output = model.forward(new_sample)
		features['feats'].append(output)
		features['paths'].append(audio_paths[idx])
	return features


if __name__ == '__main__':
	extract_features()
