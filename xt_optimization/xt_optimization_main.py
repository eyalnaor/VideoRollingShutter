import os
import sys
sys.path.append(os.path.dirname(__file__))
from tqdm import tqdm
from torch.autograd import backward, Variable
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import json
import xt_optimization_config_utils
import utils
import xt_optimization_utils
from loss import ValidityLoss, NNLoss



def main(config):
	"""
	Goes through main() only when used separately. When part of VideoRS_Main it goes straight to xt_optimization_single
	"""
	if config["video_name"] is None:
		rs_input_father=os.path.join(config["organized_data_folder"],'input_rolling',config["dataset"])
		video_names = sorted([subdir for subdir in os.listdir(rs_input_father) if os.path.isdir(os.path.join(rs_input_father, subdir))])
	else:
		if isinstance(config["video_name"],str):
			video_names=[config["video_name"]]
		elif isinstance(config["video_name"],list):
			video_names=config["video_name"]
		else: assert False
		assert isinstance(video_names, list)

	for vid_idx, video_name in enumerate(video_names):
		print(f'entered video: {video_name}, #{vid_idx+1} of {len(video_names)}')
		GT_path=os.path.join(config["organized_data_folder"],'GT',config['dataset'],video_name)

		path_to_validity = os.path.join(config["organized_data_folder"], 'AfterMergeNetResults',
											   config["dataset"], config["alignment"], video_name)
		path_to_NN = os.path.join(config["organized_data_folder"], 'input_rolling', config['dataset'], video_name)

		GT_in_name, gt_direction = GTinName_from_config(config)
		save_folder=os.path.join(config['save_dir'], video_name, 'final_result')
		xt_optimization_single(config, path_to_validity, path_to_NN, NN_in_name=None, save_folder=save_folder,
							   GT_path=GT_path, gt_in_name=GT_in_name, gt_direction=gt_direction)

	del config, video_names
	torch.cuda.empty_cache()




def xt_optimization_single(config, path_to_validity, path_to_NN, NN_in_name=None, save_folder=None, GT_path=None, gt_in_name=None, gt_direction='left'):
	"""
	extracted separate for simplicity and workability with VideoRS_main
	"""
	if save_folder is None:  # save alongsided the input
		save_folder=os.path.join(os.path.split(path_to_validity)[0],'after_xt_optimization')


	validity_tensor = utils.load_video_folder_to_torch(path_to_validity)
	_, _, _, _, canny_validity, canny_early_validity = xt_optimization_utils.canny_on_vid(validity_tensor[0, :, :, :, :],
																		  return_only_early_threshold_vid=False)
	NN_weights = process_NN_weights(config, canny_early_validity)
	NN_layer_tensor = utils.load_video_folder_to_torch(path_to_NN, must_include_in_name=NN_in_name)
	# set initial guess for result image
	result = validity_tensor.clone()
	result.requires_grad = True
	# skip video if too small, smaller than wanted patch
	skip_vid = check_if_skip_video(config, result.shape, validity_tensor.shape, NN_layer_tensor.shape)
	if skip_vid:
		print(
			f'******** video too small for this loss!! Wanted patches are larger than vid. Skipping... ********')
		del NN_layer_tensor, result, validity_tensor
		torch.cuda.empty_cache()
	# define loss layers and optimizer
	validity_weights = edit_weights(NN_weights, edit=config['validity_loss']['edit_weights'])
	validity_loss_layer = ValidityLoss(validity_tensor, config,
									   validity_patch_size=config["validity_loss"]["validity_patch_size"],
									   weights=validity_weights)
	nn_loss_layer = NNLoss(NN_layer_tensor, NN_patch_size=config["NN_loss"]["NN_patch_size"],
						   loss_name=config["NN_loss"]["name"], t_flip=config["NN_loss"]["temporal_flip"],
						   h_flip=config["NN_loss"]["hor_flip"], v_flip=False, nlist=config["faiss_nlist"], nprobe=config["faiss_nprobe"],
						   weights=NN_weights)
	if config["NN_loss_2"]["enable"]:  # want an ADDITIONAL NN loss
		nn_loss_layer_2 = NNLoss(NN_layer_tensor, NN_patch_size=config["NN_loss_2"]["NN_patch_size"],
								 loss_name=config["NN_loss_2"]["name"], t_flip=config["NN_loss_2"]["temporal_flip"],
								 h_flip=config["NN_loss_2"]["hor_flip"], v_flip=False, nlist=config["faiss_nlist"], nprobe=config["faiss_nprobe"],
								 weights=NN_weights)
	if config['optimization']['name'] == 'Adam':
		optimizer = torch.optim.Adam([result], lr=config['optimization']['lr'])
	elif config['optimization']['name'] == 'SGD':
		optimizer = torch.optim.SGD([result], lr=config['optimization']['lr'],
									momentum=config['optimization']['momentum'])
	else:
		assert False, f'unknown optimizer: {config["optimization"]["name"]}'
	# losses will be combined according to scaling_factor which will be computed in the first epoch
	scaling_factor = 1

	if GT_path is not None:
		GT = utils.load_video_folder_to_torch(GT_path, must_include_in_name=gt_in_name)
		if gt_direction == 'left':
			GT = GT[:, 1:, :, :]  # remove first
		elif gt_direction == 'center':
			GT = GT[:, 1:-1, :, :]  # remove first and last
		assert GT.shape[1:] == result.shape[1:]

	# train
	for epoch in tqdm(range(-1, config['num_epochs'])):
		# forward pass
		validity_loss = validity_loss_layer(result)

		if config["NN_loss_2"]["enable"]:
			nn_loss = (1 - config["NN_loss_2"]["weight_in_NN"]) * nn_loss_layer(result) + config["NN_loss_2"][
				"weight_in_NN"] * nn_loss_layer_2(result)
		else:
			nn_loss = 1.0 * nn_loss_layer(result)

		if config['scaling_method'] == 'no_scaling':
			scaling_factor = 1.0  # overrun scaling factor so will not be used
		loss = config['validity_loss_ration'] * scaling_factor * validity_loss + (
					1 - config['validity_loss_ration']) * nn_loss

		# calculate scaling_factor. If mergenet_result needs after first epoch, else val_loss=0.
		if epoch == 0:
			# calculate scaling_factor
			if config['scaling_method'] == "gradient_scaling":
				optimizer.zero_grad()
				backward(validity_loss)
				grad_from_validity_loss = result.grad.detach().abs().mean()
				optimizer.zero_grad()
				backward(nn_loss)
				grad_from_nn_loss = result.grad.detach().abs().mean()
				scaling_factor = grad_from_nn_loss / grad_from_validity_loss
			elif config['scaling_method'] == "loss_scaling":
				scaling_factor = nn_loss.detach() / validity_loss.detach()
			else:
				raise NotImplementedError()


		# log epoch state
		if GT_path is not None:
			psnr = utils.PSNR_video_avg_frames(result, GT)
			ssim = utils.SSIM_video_avg_frames(result, GT).item()
			if config['print_every'] is not None and not epoch % config['print_every']:
				print(
					f'**epoch {epoch}: psnr: {psnr:.3f}, ssim: {ssim:.4f}, loss:{loss:.0f}, nn_loss:{nn_loss:.0f}, validity_loss:{validity_loss:.0f}**')

		# perform training step
		optimizer.zero_grad()
		backward(loss)
		optimizer.step()
	if GT_path is not None:
		final_psnr = utils.PSNR_video_avg_frames(result, GT)
		final_ssim = utils.SSIM_video_avg_frames(result, GT).item()
		del GT

	# save final result
	utils.write_im_vid(save_folder, result)
	del NN_layer_tensor, loss, nn_loss, nn_loss_layer, result, scaling_factor, validity_loss, validity_loss_layer, optimizer, epoch, validity_tensor
	torch.cuda.empty_cache()
	if GT_path is not None:
		return final_psnr, final_ssim
	else:
		return None

def check_if_skip_video(config, result_shape, validity_tensor_shape, NN_layer_tensor_shape):
	if any([a < b for (a, b) in zip(result_shape[2:5], config['validity_loss']['validity_patch_size'])]) or any(
			[a < b for (a, b) in
			 zip(validity_tensor_shape[2:5], config['validity_loss']['validity_patch_size'])]) or any(
		[a < b for (a, b) in zip(result_shape[2:5], config['NN_loss']['NN_patch_size'])]) or any(
		[a < b for (a, b) in zip(NN_layer_tensor_shape[2:5], config['NN_loss']['NN_patch_size'])]):
		return True
	else:
		return False

def edit_weights(input_weights, edit='reversal'):
	"""assumes weights in&out are BCTHW"""
	if edit=='reversal':  #max-input, so max-->min, min-->max
		maxs=torch.amax(input_weights,dim=(3,4),keepdim=True).repeat((1,1,1,input_weights.shape[3],input_weights.shape[4]))
		edited_weights=maxs-input_weights
	elif edit=='inverse':
		edited_weights = 1/torch.clip(input_weights,0.01,input_weights.max())
	elif edit=='none':
		edited_weights=input_weights.clone()
	else: assert False, f'unknown edit in edit_weights: {edit}'
	return edited_weights

def process_NN_weights(config, weights):
	assert not (config["NN_loss"]["weighted_MSE_norm_power"] is not None and config["NN_loss"]["weighted_MSE_cutoff"] is not None), f'at least one of weighted_MSE_norm_power or weighted_MSE_cutoff needs to be None, cant operate together'
	if config["NN_loss"]["weighted_MSE_cutoff"] is not None:
		weights = torch.clip(weights, 0.0, config['NN_loss']['weighted_MSE_cutoff'])
	if config["NN_loss"]["weighted_MSE_norm_power"] is not None:
		weights = ((weights-weights.min())/(weights.max()-weights.min()))**config["NN_loss"]["weighted_MSE_norm_power"]
	if config["NN_loss"]["weighted_MSE_dilation"] is not None:
		m=nn.MaxPool2d(config["NN_loss"]["weighted_MSE_dilation"], stride=1, padding=config["NN_loss"]["weighted_MSE_dilation"]//2)
		weights=weights.squeeze().permute(1,0,2,3)  # from [1CTHW] to [TCHW], needed for 2d pooling
		weights=m(weights)
		weights=weights.permute(1,0,2,3).unsqueeze(0)
	return weights

def GTinName_from_config(config):
	if config['dataset'] == 'carla' and config['alignment'] == 'left':
		GT_in_name = 'gs_f.png'
		gt_direcion='left'
	elif config['dataset'] == 'carla' and config['alignment'] == 'center':
		GT_in_name = 'gs_m.png'
		gt_direcion = 'center'
	elif config['dataset'] == 'fastec' and config['alignment'] == 'left':
		GT_in_name = 'first.png'
		gt_direcion = 'left'
	elif config['dataset'] == 'fastec' and config['alignment'] == 'center':
		GT_in_name = 'middle.png'
		gt_direcion = 'center'
	elif config['dataset'] == 'spinners' and config['alignment'] == 'center':
		GT_in_name = 'center.png'
		gt_direcion = 'center'
	elif config['dataset'] == 'spinners' and config['alignment'] == 'left':
		GT_in_name = 'left.png'
		gt_direcion = 'left'
	elif config['dataset'] == 'youtube' and config['alignment'] == 'center':
		GT_in_name = 'center.png'
		gt_direcion = 'center'
	elif config['dataset'] == 'youtube' and config['alignment'] == 'left':
		GT_in_name = 'left.png'
		gt_direcion = 'left'
	elif config['dataset'] == 'youtube2' and config['alignment'] == 'center':
		GT_in_name = 'center.png'
		gt_direcion = 'center'
	elif config['dataset'] == 'youtube2' and config['alignment'] == 'left':
		GT_in_name = 'left.png'
		gt_direcion = 'left'
	else:
		assert False
	return GT_in_name, gt_direcion

def extract_dataset_from_vid_path(video_name):
	dataset = None
	if 'fastec_test' in video_name:
		assert dataset is None
		dataset = 'fastec_test'
	if 'carla_test' in video_name:
		assert dataset is None
		dataset = 'carla_test'
	assert dataset is not None
	return dataset


if __name__ == "__main__":
	# Need to generate config if running only xt_optimization, else gets config externally.
	parser = xt_optimization_config_utils.create_parser()
	args = parser.parse_args()
	config = xt_optimization_config_utils.startup(json_path=args.config, args=args)
	main(config)

