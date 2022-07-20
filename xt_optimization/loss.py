import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import faiss
import faiss.contrib.torch_utils
from xt_optimization import fold3d, fold_utils
from matplotlib import pyplot as plt

def calc_dist_l2(X, Y):
	"""
	Calculate distances between patches
	:param X: tensor of n patches of size k
	:param Y: tensor of m patches of size k
	:return: l2 distance matrix - tensor of shape n * m
	"""
	Y = Y.transpose(0, 1)
	X2 = X.pow(2).sum(1, keepdim=True)
	Y2 = Y.pow(2).sum(0, keepdim=True)
	XY = X @ Y
	return X2 - (2 * XY) + Y2


def preprocess_patches_for_faiss(KV_patches, nlist=None, nprobe=None):
	"""
	:param KV_patches: input as torch tensor [1,d,N]
	:param nlist: if not None, number of bins to divide the Keys into
	:param nprobe: if not None, the number of bins to search in for Queries (bins with closest centroids)
	:return: faiss IndexFlatL2 object, able to be searched
	"""
	d=KV_patches.shape[1]
	N=KV_patches.shape[2]

	KV_patches_for_faiss = KV_patches.squeeze().transpose(1, 0).contiguous()
	res = faiss.StandardGpuResources()
	if nlist is None:  # standard flat index, no bins
		index = faiss.GpuIndexFlatL2(res, d)
	else:
		nlist_ = nlist  # checked is not None
		nprobe_ = nprobe if nprobe is not None else nlist_//4  # from faiss' git, 1/4 gives over 90% recall
		assert nprobe_ >= 1
		quantizer = faiss.IndexFlatL2(d)  # used to find which bucket to search in (nearest centroid)
		quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)
		cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist_)
		index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
		assert not index.is_trained
		index.train(KV_patches_for_faiss)
		assert index.is_trained
		index.nprobe = nprobe_

	index.add(KV_patches_for_faiss)  # add vectors to the index
	assert index.ntotal == N
	assert index.d == d

	return index



class NNLoss(nn.Module):

	def __init__(self, valid_video, NN_patch_size=(1, 5, 5), loss_name='MSE', t_flip=False, h_flip=False, v_flip=False, nlist=None, nprobe=None, weights=None):
		super(NNLoss, self).__init__()

		self.NN_patch_size = NN_patch_size  # 3d patch size of THW
		self.t_flip=t_flip
		self.h_flip=h_flip
		self.v_flip=v_flip
		self.loss_name=loss_name
		if weights is not None:
			self.weights_patches_2d = self.extract_weight_patches(weights, mean_patches=True, sqrt=True, add_eps=True)
		self.NN_loss_fn = self.define_loss()

		valid_patches_2d, valid_patches_8d_size, valid_patches_8d_ndim = self.extract_valid_patches_2d(valid_video)



		self.valid_patches_8d_size = valid_patches_8d_size  # save aside the shape if will want to fold back
		self.valid_patches_8d_ndim = valid_patches_8d_ndim
		self.valid_patches_2d=valid_patches_2d.squeeze(0).permute(1, 0)  # needed as V
		self.faiss_idx_valid_patches = preprocess_patches_for_faiss(valid_patches_2d, nlist=nlist, nprobe=nprobe)


	class weighted_MSE(nn.Module):
		def __init__(self, weights):
			super(NNLoss.weighted_MSE, self).__init__()
			self.weights = weights

		def forward(self, out, target):
			return torch.mean(torch.square(out * self.weights-target * self.weights))
	def extract_valid_patches_2d(self, valid_video, enable_flips=True):
		valid_patches = fold3d.unfold3d(valid_video, self.NN_patch_size, stride=1, use_padding=False)
		valid_patches_2d, valid_patches_8d_size, valid_patches_8d_ndim = fold_utils.view_as_2d(valid_patches)
		valid_patches_2d = valid_patches_2d.transpose(1, 0).unsqueeze(0)
		if enable_flips:  # useful for patch "bank", not weights for instance
			if self.t_flip:
				valid_video_auged = valid_video.flip([2])  # temporal flip (on top of spatial)
				valid_patches_auged = fold3d.unfold3d(valid_video_auged, self.NN_patch_size, stride=1,
													  use_padding=False)
				valid_patches_2d_auged, _, _ = fold_utils.view_as_2d(valid_patches_auged)
				valid_patches_2d_auged = valid_patches_2d_auged.transpose(1, 0).unsqueeze(0)
				valid_patches_2d = torch.cat((valid_patches_2d, valid_patches_2d_auged), dim=2)


			if self.h_flip:
				valid_video_auged=valid_video.flip([4]) # hor flip
				valid_patches_auged = fold3d.unfold3d(valid_video_auged, self.NN_patch_size, stride=1, use_padding=False)
				valid_patches_2d_auged, _, _ = fold_utils.view_as_2d(valid_patches_auged)
				valid_patches_2d_auged = valid_patches_2d_auged.transpose(1, 0).unsqueeze(0)
				valid_patches_2d=torch.cat((valid_patches_2d,valid_patches_2d_auged),dim=2)
				if self.t_flip:
					valid_video_auged = valid_video_auged.flip([2]) # temporal flip (on top of spatial)
					valid_patches_auged = fold3d.unfold3d(valid_video_auged, self.NN_patch_size, stride=1,
														  use_padding=False)
					valid_patches_2d_auged, _, _ = fold_utils.view_as_2d(valid_patches_auged)
					valid_patches_2d_auged = valid_patches_2d_auged.transpose(1, 0).unsqueeze(0)
					valid_patches_2d = torch.cat((valid_patches_2d, valid_patches_2d_auged), dim=2)

		return valid_patches_2d, valid_patches_8d_size, valid_patches_8d_ndim

	def extract_weight_patches(self, weights, normalize=True, mean_patches=True, sqrt=True, add_eps=True):
		if normalize:
			weights=(weights-weights.min())/(weights.max()-weights.min())
		weights_patches_2d, _, _ = self.extract_valid_patches_2d(weights, enable_flips=False)
		weights_patches_2d = weights_patches_2d.squeeze(0).permute(1, 0)  # needed as V
		if mean_patches:
			weights_patches_2d=weights_patches_2d.mean(dim=1,keepdim=True).repeat([1,weights_patches_2d.shape[1]])
		if sqrt:
			weights_patches_2d = torch.sqrt(weights_patches_2d)
		if add_eps:
			weights_patches_2d = torch.clip(weights_patches_2d + 0.01, 0.0, 1.0)
		return weights_patches_2d

	def define_loss(self):
		if self.loss_name == 'MSE':
			return torch.nn.MSELoss(reduction='sum')
		elif self.loss_name == 'weighted_MSE':
			return self.weighted_MSE(self.weights_patches_2d)
		elif self.loss_name == 'L1':
			return torch.nn.L1Loss()
		else:
			assert False, f'assertion error in define_opt(), loss does not exist, is {self.loss_name}'

	def forward(self, result):

		result_patches = fold3d.unfold3d(result, self.NN_patch_size, stride=1, use_padding=False)
		result_patches_2d, result_patches_8d_size, result_patches_8d_ndim = fold_utils.view_as_2d(result_patches)
		result_patches_2d=result_patches_2d.contiguous()

		patches_per_iter=50000
		iters_needed = (result_patches_2d.shape[0] - 1) // patches_per_iter + 1
		D, I = None, None
		for iter in range(iters_needed):
			D_, I_ = self.faiss_idx_valid_patches.search(result_patches_2d[iter * patches_per_iter:(iter + 1) * patches_per_iter, :], 1)
			if D is None:
				D = torch.clone(D_)
				I = torch.clone(I_)
			else:
				D=torch.cat((D,D_),dim=0)
				I=torch.cat((I,I_),dim=0)

		I=I.squeeze()
		NNed_patches = F.embedding(I, self.valid_patches_2d)

		loss = self.NN_loss_fn(result_patches_2d, NNed_patches)
		return loss

	def save_NN_dist(self, NNed_patches, result, result_patches_8d_ndim, result_patches_8d_size):
		zzz = fold_utils.view_2d_as(NNed_patches, result_patches_8d_size, result_patches_8d_ndim)
		NNed_patches_folded = fold3d.fold3d(zzz, stride=1, use_padding=False, reduce='mean', std=1.7, dists=None,
											external_weights=None)
		diff_to_NN = torch.mean(torch.abs(result - NNed_patches_folded), dim=1, keepdim=True)
		diff_to_NN_folder = f'{self.config["save_dir"]}/NN_dist'
		if not os.path.isdir(diff_to_NN_folder):  # want to save only once
			os.makedirs(diff_to_NN_folder, exist_ok=False)  # want to save only once
			stretched_diff_to_NN = diff_to_NN / (diff_to_NN.max(dim=3, keepdim=True)[0].max(dim=4, keepdim=True)[0])
			image.write_im_vid(diff_to_NN_folder, stretched_diff_to_NN)
			diff_to_NN_folder_with_colorbar = f'{self.config["save_dir"]}/NN_dist_colorbar'
			os.makedirs(diff_to_NN_folder_with_colorbar, exist_ok=False)  # want to save only once

			for f_idx in range(diff_to_NN.shape[2]):
				im_npy = diff_to_NN[0, 0, f_idx, :, :].detach().cpu().numpy()
				im_save_path = f'{diff_to_NN_folder_with_colorbar}/{f_idx:05d}.png'
				plt.figure()
				plt.imshow(im_npy)
				plt.colorbar()
				plt.savefig(im_save_path)
				plt.close()

class ValidityLoss(nn.Module):

	def __init__(self, validity_tensor, config, validity_patch_size=(1,5,5), weights=None):
		super(ValidityLoss, self).__init__()
		self.config = config
		self.validity_patch_size = validity_patch_size  # 3d patch size of THW
		if weights is not None:
			self.weights_patches_2d = self.extract_weight_patches(weights, mean_patches=True, sqrt=True, add_eps=True)
		self.validity_loss_fn=self.define_loss()
		self.validity_tensor = validity_tensor

	class weighted_MSE(nn.Module):
		def __init__(self, weights):
			super(ValidityLoss.weighted_MSE, self).__init__()
			self.weights = weights.permute(1,0).unsqueeze(0)

		def forward(self, out, target):
			return torch.mean(torch.square(out * self.weights-target * self.weights))

	def extract_valid_patches_2d(self, valid_video):
		valid_patches = fold3d.unfold3d(valid_video, self.validity_patch_size, stride=1, use_padding=False)
		valid_patches_2d, valid_patches_8d_size, valid_patches_8d_ndim = fold_utils.view_as_2d(valid_patches)
		valid_patches_2d = valid_patches_2d.transpose(1, 0).unsqueeze(0)
		return valid_patches_2d, valid_patches_8d_size, valid_patches_8d_ndim

	def extract_weight_patches(self, weights, normalize=True, mean_patches=True, sqrt=True, add_eps=True):
		if normalize:
			weights=(weights-weights.min())/(weights.max()-weights.min())
		weights_patches_2d, _, _ = self.extract_valid_patches_2d(weights)
		weights_patches_2d = weights_patches_2d.squeeze(0).permute(1, 0)  # needed as V
		if mean_patches:
			weights_patches_2d=weights_patches_2d.mean(dim=1,keepdim=True).repeat([1,weights_patches_2d.shape[1]])
		if sqrt:
			weights_patches_2d = torch.sqrt(weights_patches_2d)
		if add_eps:
			weights_patches_2d = torch.clip(weights_patches_2d + 0.01, 0.0, 1.0)
		return weights_patches_2d

	def define_loss(self):
		loss_name = self.config['validity_loss']['name']
		if loss_name == 'MSE':
			return torch.nn.MSELoss(reduction='sum')
		elif loss_name == 'weighted_MSE':
			return self.weighted_MSE(self.weights_patches_2d)
		elif loss_name == 'L1':
			return torch.nn.L1Loss()
		else:
			assert False, f'assertion error in define_opt(), loss does not exist, is {loss_name}'

	def forward(self, result):

		assert self.validity_patch_size[0] == 1, f'for memory reasons need to split the work, and here doing so frame-by-frame'
		result_patches=None
		best_aug_patches=None
		for f_idx in range(result.shape[2]):
			result_frame=result[:,:,f_idx:f_idx+1,:,:]
			result_frame_patches_2d, result_patches_8d_size, result_patches_8d_ndim = fold_utils.view_as_2d(
				fold3d.unfold3d(result_frame, self.validity_patch_size, stride=1, use_padding=False))
			result_frame_patches_2d = result_frame_patches_2d.transpose(1, 0).unsqueeze(0)

			validity_tensor_frame=self.validity_tensor[:,:,f_idx:f_idx+1,:,:]
			validity_tensor_frame_patches_3d,_ = fold_utils.view_8d_as_3d(fold3d.unfold3d(validity_tensor_frame, self.validity_patch_size, stride=1, use_padding=False))  #preserves batch separately

			best_aug_patches_frame=validity_tensor_frame_patches_3d

			if result_patches is None:
				result_patches = result_frame_patches_2d
				best_aug_patches=best_aug_patches_frame
			else:
				result_patches = torch.cat((result_patches,result_frame_patches_2d),dim=2)
				best_aug_patches = torch.cat((best_aug_patches, best_aug_patches_frame),dim=2)
		loss = self.validity_loss_fn(result_patches, best_aug_patches)

		return loss
