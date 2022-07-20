import os
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
import kornia as K

try:
    from torch.utils.tensorboard import SummaryWriter
    no_summary=False
except:
    no_summary=True
from tqdm.autonotebook import tqdm
import utils
from tqdm import trange
import sys
import utils


class mergenet:  # The base network
    def __init__(self, config, device):
        self.config = config
        self.device=device
        self.aug_num = 16  # affects size of input and output
        self.size_in = self.aug_num*3  # 3 for RGB
        if self.config['output_type'] == "weights":
            self.size_out=self.aug_num  # single weight for each augmentation, for all RGB
        elif self.config['output_type'] == "image":
            self.size_out = 3  # RGB

        self.hidden_width = self.config['architecture']['hidden_width']
        self.hidden_depth = self.config['architecture']['hidden_depth']
        self.conv_kernel_size=self.config['architecture']['conv_kernel_size']
        self.conv_padding=[i//2 for i in self.conv_kernel_size]
        self.net = self.build_network()
        self.optimizer = self.define_opt() if hasattr(config, 'optimization') else None

        self.norm_minus1_1_for_loss_only=False  # If loss will require normalizing to [-1,1] will change here to True in define_loss()
        self.loss_fn = self.define_loss() if hasattr(config, 'loss') else None
        # to run tensorboard run the following command in the relevant results folder
        # tensorboard --logdir logs_dir
        if self.config['save_dir'] is not None:
            if not no_summary and config['need_to_train']:  #check if can and if needed to save training logs
                self.writer = SummaryWriter(os.path.join(self.config['save_dir'], 'logs_dir'))

        # total number of epochs
        self.epochs = self.config['num_epochs'] if hasattr(self.config, 'num_epochs') else 1
        # current or start epoch number
        self.epoch = 0

        self.scheduler = self.define_lr_sched() if hasattr(config,'lr_sched') else None

    def build_network(self):  # BASE version. Other modes override this function
        """
        take the network flag or parameters from config and create network
        :return: net - a torch class/object that can be trained
        """
        """
        ZSSR architecture: [[3, 3, 3, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 64],
                            [3, 3, 64, 3]]
        """

        if self.config['architecture']['activation'] == "ReLU":
            activation = nn.ReLU()
        else: assert False

        layers = [nn.Conv2d(in_channels=self.size_in, out_channels=self.hidden_width, kernel_size=self.conv_kernel_size,
                            padding=tuple(self.conv_padding), padding_mode='reflect'), activation]
        for hidden_layer in range(self.hidden_depth):
            layers.append(nn.Conv2d(in_channels=self.hidden_width, out_channels=self.hidden_width, kernel_size=self.conv_kernel_size,
                            padding=tuple(self.conv_padding), padding_mode='reflect'))
            layers.append(activation)
        layers.append(nn.Conv2d(in_channels=self.hidden_width, out_channels=self.size_out, kernel_size=self.conv_kernel_size,
                            padding=tuple(self.conv_padding), padding_mode='reflect'))

        if self.config["output_type"]=='weights':
            layers.append(nn.Softmax(dim=1))
        net = nn.Sequential(*layers).to(self.device)
        return net

    class laplacian_loss(nn.Module):
        def __init__(self, config, laplacian_dist="MSE", laplacian_kernel_size=3, additional_loss='MSE', MSE_reduction="mean", laplacian_weight=0.999, normalize_laplacian=False):
            super(mergenet.laplacian_loss, self).__init__()
            self.config=config
            self.laplacian_dist=laplacian_dist
            if self.laplacian_dist == 'MSE':
                self.lap_loss=torch.nn.MSELoss(reduction=MSE_reduction)
            elif self.laplacian_dist=='L1':
                self.lap_loss=torch.nn.L1Loss()
            else: assert False, f'unknown loss function for laplacian_dist: {laplacian_dist}'
            self.laplacian_kernel_size=laplacian_kernel_size
            self.additional_loss = additional_loss
            if self.additional_loss == 'MSE':
                self.loss2=torch.nn.MSELoss(reduction=MSE_reduction)
            elif self.additional_loss=='L1':
                self.loss2=torch.nn.L1Loss()
            else: assert False, f'unknown loss function for additional_loss: {additional_loss}'
            self.laplacian_weight = laplacian_weight
            self.normalize_laplacian = normalize_laplacian


        def forward(self, out, target):
            laplacian_out=K.filters.laplacian(out, kernel_size=self.laplacian_kernel_size)
            laplacian_target=K.filters.laplacian(target, kernel_size=self.laplacian_kernel_size)
            if self.normalize_laplacian:  #normalize by target nums for both
                target_min, target_max=torch.min(laplacian_target) , torch.max(laplacian_target)
                laplacian_out = (laplacian_out - target_min) / (target_max - target_min)
                laplacian_target = (laplacian_target - target_min) / (target_max - target_min)
            loss = self.laplacian_weight * self.lap_loss(laplacian_out, laplacian_target) + (
                        1 - self.laplacian_weight) * self.loss2(out, target)
            return loss

    def define_loss(self):
        loss_name = self.config['loss']['name']
        if loss_name == 'MSE':
            return torch.nn.MSELoss(reduction=self.config["loss"]["MSE_reduction"])
        elif loss_name == 'L1':
            return torch.nn.L1Loss()
        elif loss_name == 'laplacian':
            return self.laplacian_loss(self.config,
                                       laplacian_dist=self.config["loss"]["laplacian_params"]["laplacian_dist"],
                                       laplacian_kernel_size=self.config["loss"]["laplacian_params"]["laplacian_kernel_size"],
                                       additional_loss=self.config["loss"]["laplacian_params"]["additional_loss"],
                                       MSE_reduction=self.config["loss"]["laplacian_params"]["MSE_reduction"],
                                       laplacian_weight=self.config["loss"]["laplacian_params"]["laplacian_weight"],
                                       normalize_laplacian=self.config["loss"]["laplacian_params"]["normalize_laplacian"])
        else:
            assert False, f'assertion error in define_opt(), loss does not exist, is {loss_name}'

    def define_opt(self):
        opt_name = self.config['optimization']['name']
        learning_rate = self.config['optimization']['lr']
        if opt_name == 'SGD':
            momentum = self.config['optimization']['SGD_momentum']
            return torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)
        elif opt_name == 'Adam':
            return torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            assert False, f'assertion error in define_opt(), optimizer does not exist, is {opt_name}'

    def define_lr_sched(self):
        gamma = self.config['lr_sched']['params']['gamma']
        milestones = self.config['lr_sched']['params']['milestones']
        step_size = self.config['lr_sched']['params']['step_size']

        if self.config['lr_sched']['name'] == 'MultiStepLR':
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.config['lr_sched']['name'] == 'StepLR':
            return lr_scheduler.StepLR(self.optimizer, step_size=int(self.epochs * step_size), gamma=gamma)

        elif self.config['lr_sched']['name'] == 'MultiStepLR_percentages':
            milestones_from_percentages = [int(i*self.config['num_epochs']) for i in self.config['lr_sched']['params']['milestones_percentages']]
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones_from_percentages, gamma=gamma)
        else:
            print('****************** NO LR_SCHED DEFINED SETTING DEFAULT *****************************')
            return lr_scheduler.StepLR(self.optimizer, step_size=self.epochs // 10, gamma=1 / 1.5)


    def forward(self, input, debug_save_mean_path=None):
        output = self.net(input)
        if self.config['output_type'] == "weights":
            # Since the net predicts the weights of each input augmentation, we need to first use it to get the predicted image, and then calc loss from gt
            output = self.image_from_weights(output, input)
        elif self.config['output_type'] == "image":
            if self.config['image_resudual']:
                mean=input.view((input.shape[0],input.shape[1]//3, 3, *input.shape[2:])).mean(dim=1)
                output = output + mean
                if debug_save_mean_path is not None:
                    print(f'saving mean for debugging!')
                    utils.write_im_vid(debug_save_mean_path, mean.transpose(dim0=0, dim1=1).unsqueeze(0))
        else: assert False, f'unknown output_type: {self.config["output_type"]}'

        return output

    def calc_loss(self, prediction, gt, calc_psnrssim=False):
        """
        calc loss according to the flags in config
        :param prediction: the output from the net.
        :param gt: the gt
        :return: the loss
        """

        if calc_psnrssim:  #want to also calculate the PSNR and SSIM, do so before normalizing
            psnr, ssim = utils.PSNR_SSIM_video_avg_frames(prediction.transpose(dim0=0,dim1=1).unsqueeze(0), gt.transpose(dim0=0,dim1=1).unsqueeze(0))
        else:
            psnr, ssim = None, None  # not used, so no need in calculating

        if self.norm_minus1_1_for_loss_only:
            prediction = 2 * prediction - 1
            gt = 2 * gt - 1
        loss = self.loss_fn(prediction, gt)
        if len(loss.shape)>0:
            loss=loss.squeeze().mean()
        return loss, psnr, ssim

    def image_from_weights(self, weights, input_augs):
        input_augs_aug_RGB = input_augs.view((input_augs.shape[0],input_augs.shape[1] // 3, 3, *input_augs.shape[2:]))
        weights_RGB=torch.unsqueeze(weights,dim=2)
        out_image=torch.mul(input_augs_aug_RGB,weights_RGB)
        out_image=torch.sum(out_image,dim=1)
        return out_image

    def train(self, data_loader_object):
        """
        :param data_loader_object: data_handler object that holds the video tensor and can make all necessary augmentations
        :return: train_logs. loss vectors for each epoch
        """
        with tqdm(total=len(data_loader_object) * self.config['num_epochs']) as pbar:
            for e in range(1, self.config['num_epochs']):
                t = time.time()
                np.random.seed()
                self.optimizer.zero_grad()
                if not e % self.config["val_every"] and not self.config['skip_val']:
                    avg_val_loss, avg_val_psnr, avg_val_ssim = self.validation(data_loader_object.dataset, e)
                if not self.config['debug'] and not e%self.config["save_model_every"]:
                    self.save_model(epoch=e, overwrite=False)

                it = 0
                for idx, (input, gt) in enumerate(data_loader_object):
                    prediction = self.forward(input.to(self.device))
                    loss, psnr, ssim = self.calc_loss(prediction, gt)
                    it += 1
                    pbar.update(1)

                if not e%50:
                    print(f'\t\tepoch:{e}, loss:{loss.item():.7f}. Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')

                # finished epoch - update and write
                loss.backward()
                prev_lr = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.scheduler.step()
                if self.optimizer.param_groups[0]["lr"]!=prev_lr:
                    print(f'***** at epoch {e}, LR changed from {prev_lr} to {self.optimizer.param_groups[0]["lr"]} *****')
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('loss', loss.item(), e)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], e)

            # save final trained model as well
            if not self.config['debug']:
                self.save_model(epoch=e, overwrite=False)

            #final validation - save entire video
            if not self.config['skip_val']:
                final_avg_val_loss, final_avg_val_psnr, final_avg_val_ssim = self.save_final_val_results(
                    data_loader_object.dataset)

            # to run tensorboard run the following command in the relevant results folder
            # tensorboard --logdir logs_dir
            if hasattr(self, 'writer'):
                self.writer.close()
            return

    def fine_tune(self, fine_tune_data_loader_object):
        """
        fine tunes model.
        Does not validate or saves intermediate models
        """
        if self.config['fine_tuning']['finetuning_lr'] is not None:   #reset the LR to what is instructed
            self.optimizer.param_groups[0]["lr"]=self.config['fine_tuning']['finetuning_lr']

        with tqdm(total=len(fine_tune_data_loader_object) * self.config['fine_tuning']['finetuning_epochs']) as pbar:
            for e in range(1, self.config['fine_tuning']['finetuning_epochs']):
                t = time.time()
                np.random.seed()
                self.optimizer.zero_grad()
                # iterations per epochs
                it = 0

                for idx, (input, gt) in enumerate(fine_tune_data_loader_object):  #getitem will bring from fine_tune since gave it editted config
                    prediction = self.forward(input.to(self.device))
                    loss, psnr, ssim = self.calc_loss(prediction, gt)
                    it += 1
                    pbar.update(1)

                if not e%50:
                    print(f'\t\tepoch:{e}, loss:{loss.item():.7f}. Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')

                # finished epoch - update and write
                loss.backward()
                self.optimizer.step()
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('loss', loss.item(), e)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], e)

            # save final trained model as well
            if not self.config['debug']:
                self.save_model(epoch=e, overwrite=False, fine_tune=True)

            # to run tensorboard run the following command in the relevant results folder
            # tensorboard --logdir logs_dir
            if hasattr(self, 'writer'):
                self.writer.close()
            return

    def eval_on_single(self, eval_single_folder, save_folder=None, must_include_in_name=None, gt_single_folder=None, gt_in_name=None, gt_direction='left'):
        """
        helpful for VideoRS_main, that works 1 video at a time
        gt_single_folder: to enable getting measurements when possible
        gt_direction: 'none', 'left' or 'center'. Since gt often has additional frames than can be reconstructed, this param helps remove wanted frame/s
        """
        if save_folder is None:
            save_folder=os.path.join(os.path.split(eval_single_folder)[0],'after_MergeNet')
        to = {'device': self.device, 'dtype': torch.float32}
        augmentations = sorted([subdir for subdir in os.listdir(eval_single_folder) if
                                os.path.isdir(os.path.join(eval_single_folder, subdir))])
        aug0_path = os.path.join(eval_single_folder,augmentations[0])
        aug0_vid = utils.load_video_folder_to_torch(aug0_path, print_=False, must_include_in_name=must_include_in_name)
        input_vid = torch.zeros((len(augmentations), *aug0_vid.shape[1:]), **to)
        input_vid[0, ...] = aug0_vid
        for aug_idx, aug in enumerate(augmentations[1:]):
            aug_path = os.path.join(eval_single_folder, aug)
            aug_vid = utils.load_video_folder_to_torch(aug_path, print_=False,
                                                       must_include_in_name=must_include_in_name)
            input_vid[aug_idx + 1, ...] = aug_vid  # +1 since 0 is inside already
        input_vid = input_vid.view(
            (-1, *input_vid.shape[2:]))  # merge the augmentations & RGB channels to a single channels dimension

        net_vid_prediction = self.eval_entire_vid(input_vid)

        utils.write_im_vid(save_folder, net_vid_prediction)

        if gt_single_folder is not None:
            gt_vid = utils.load_video_folder_to_torch(gt_single_folder, print_=False, must_include_in_name=gt_in_name)[
                0, ...]  # the [0,...] is to lose the batch dimension
            if gt_direction == 'left':
                gt_vid = gt_vid[:, 1:, :, :]  # remove first
            elif gt_direction == 'center':
                gt_vid = gt_vid[:, 1:-1, :, :]  # remove first and last
            assert gt_vid.shape[1:] == input_vid.shape[1:]

            loss, psnr, ssim = self.calc_loss(net_vid_prediction.transpose(dim0=0, dim1=1),
                                                                      gt_vid.transpose(dim0=0, dim1=1),
                                                                      calc_psnrssim=True)
            return psnr, ssim
        else:
            return None, None

    def results_on_TestSet(self, config_, save_results=False, debug_save_mean_path=None):
        # load one video at a time to avoid memory issues
        losses_for_mean, psnrs_for_mean, ssims_for_mean, losses_for_net, psnrs_for_net, ssims_for_net=[],[],[],[],[],[]
        for dataset_idx, (input_father_folder, input_father_in_name, gt_father_folder, gt_father_in_name) in enumerate(zip(
                config_["test_set"]["test_input_father_dir"], config_["training_input_in_name"],
                config_["test_set"]["test_gt_father_dir"], config_["training_gt_in_name"])):  # iterate over datasets
            dataset_name = config_['datasets'][dataset_idx] if config_['datasets'] is not None else ''
            losses_for_mean_single, psnrs_for_mean_single, ssims_for_mean_single, losses_for_net_single, psnrs_for_net_single, ssims_for_net_single = [], [], [], [], [], []
            augmentations=sorted([subdir for subdir in os.listdir(input_father_folder) if os.path.isdir(os.path.join(input_father_folder, subdir))])
            aug_example=os.path.join(input_father_folder,augmentations[0])
            vids=sorted([subdir for subdir in os.listdir(aug_example) if os.path.isdir(os.path.join(aug_example, subdir))])
            to = {'device': self.device, 'dtype': torch.float32}

            for idx,vid_name in enumerate(vids):
                print(f'in results_on_TestSet, vid #{idx+1} out of {len(vids)}')

                #load aug0 first to get shape instead of reasigning after each aug
                in_GS_made = True if "GS_made" in os.listdir(f'{input_father_folder}/{augmentations[0]}/{vid_name}') else False
                aug0_path = f'{input_father_folder}/{augmentations[0]}/{vid_name}{"/GS_made" if in_GS_made else ""}'
                aug0_vid=utils.load_video_folder_to_torch(aug0_path, print_=False, must_include_in_name=input_father_in_name)
                input_vid=torch.zeros((len(augmentations),*aug0_vid.shape[1:]),**to)
                input_vid[0,...]=aug0_vid
                for aug_idx, aug in enumerate(augmentations[1:]):
                    in_GS_made = True if "GS_made" in os.listdir(f'{input_father_folder}/{aug}/{vid_name}') else False
                    aug_path = f'{input_father_folder}/{aug}/{vid_name}{"/GS_made" if in_GS_made else ""}'
                    aug_vid = utils.load_video_folder_to_torch(aug_path, print_=False,must_include_in_name=input_father_in_name)
                    input_vid[aug_idx+1, ...] = aug_vid  # +1 since 0 is inside already
                input_vid = input_vid.view((-1, *input_vid.shape[2:]))  # merge the augmentations & RGB channels to a single channels dimension

                gt_path=os.path.join(gt_father_folder,vid_name)
                gt_vid=utils.load_video_folder_to_torch(gt_path, print_=False, must_include_in_name=gt_father_in_name)[0,...]  # the [0,...] is to lose the batch dimension
                if input_father_in_name == 'left':
                    gt_vid=gt_vid[:,1:,:,:]  # remove first
                elif input_father_in_name == 'center':
                    gt_vid=gt_vid[:,1:-1,:,:]  # remove first and last
                else: assert False
                assert gt_vid.shape[1:]==input_vid.shape[1:]

                # loaded input_vid and gt_vid - now check both mean and net's results
                vid_in = input_vid.view((input_vid.shape[0] // 3, 3, *input_vid.shape[1:]))
                mean_of_augs = vid_in.mean(dim=0)
                loss_for_mean, psnr_for_mean, ssim_for_mean = self.calc_loss(mean_of_augs.transpose(dim0=0, dim1=1), gt_vid.transpose(dim0=0, dim1=1), calc_psnrssim=True)

                losses_for_mean_single.append(loss_for_mean.item()); psnrs_for_mean_single.append(psnr_for_mean); ssims_for_mean_single.append(ssim_for_mean)
                if debug_save_mean_path is not None:
                    debug_save_mean_path_vid=os.path.join(debug_save_mean_path,vid_name)
                else:
                    debug_save_mean_path_vid=None
                net_vid_prediction = self.eval_entire_vid(input_vid,debug_save_mean_path=debug_save_mean_path_vid)

                loss_for_net, psnr_for_net, ssim_for_net = self.calc_loss(net_vid_prediction.transpose(dim0=0, dim1=1),
                                                                             gt_vid.transpose(dim0=0, dim1=1),
                                                                             calc_psnrssim=True)
                print(f'in results_on_TestSet, vid {vid_name}. PSNRs: mean {psnr_for_mean:.03f}, net: {psnr_for_net:.04f}, SSIMs: mean {ssim_for_mean:.03f}, net: {ssim_for_net:.04f}')
                losses_for_net_single.append(loss_for_net.item()); psnrs_for_net_single.append(psnr_for_net); ssims_for_net_single.append(ssim_for_net)
                if save_results and not config_['debug']:
                    test_save_path = os.path.join(config_['save_dir'], 'test_set_final', f'dataset{dataset_idx}', vid_name)
                    utils.write_im_vid(test_save_path, net_vid_prediction)

            avg_loss_for_mean_single = sum(losses_for_mean_single) / len(losses_for_mean_single); avg_psnr_for_mean_single = sum(psnrs_for_mean_single) / len(psnrs_for_mean_single); avg_ssim_for_mean_single = sum(ssims_for_mean_single) / len(ssims_for_mean_single)
            avg_loss_for_net_single = sum(losses_for_net_single) / len(losses_for_net_single); avg_psnr_for_net_single = sum(psnrs_for_net_single) / len(psnrs_for_net_single); avg_ssim_for_net_single = sum(ssims_for_net_single) / len(ssims_for_net_single)



            print(
                f'\t\tMeasures for test set of {dataset_name}, for naive mean of augs: loss:{avg_loss_for_mean_single:.5f}, PSNR: {avg_psnr_for_mean_single:.5f}, SSIM: {avg_ssim_for_mean_single:.6f}')
            print(
                f'\t\tMeasures for test set of {dataset_name}, for net output: loss:{avg_loss_for_net_single:.5f}, PSNR: {avg_psnr_for_net_single:.5f}, SSIM: {avg_ssim_for_net_single:.6f}')
            losses_for_mean=losses_for_mean+losses_for_mean_single; psnrs_for_mean=psnrs_for_mean+psnrs_for_mean_single; ssims_for_mean=ssims_for_mean+ssims_for_mean_single
            losses_for_net=losses_for_net+losses_for_net_single; psnrs_for_net=psnrs_for_net+psnrs_for_net_single; ssims_for_net=ssims_for_net+ssims_for_net_single

        avg_loss_for_mean = sum(losses_for_mean) / len(losses_for_mean)
        avg_psnr_for_mean = sum(psnrs_for_mean) / len(psnrs_for_mean)
        avg_ssim_for_mean = sum(ssims_for_mean) / len(ssims_for_mean)
        avg_loss_for_net = sum(losses_for_net) / len(losses_for_net)
        avg_psnr_for_net = sum(psnrs_for_net) / len(psnrs_for_net)
        avg_ssim_for_net = sum(ssims_for_net) / len(ssims_for_net)

        print(f'\t\tMeasures for AVERAGE test set for naive mean of augs: loss:{avg_loss_for_mean:.5f}, PSNR: {avg_psnr_for_mean:.5f}, SSIM: {avg_ssim_for_mean:.6f}')
        print(f'\t\tMeasures for AVERAGE test set for net output: loss:{avg_loss_for_net:.5f}, PSNR: {avg_psnr_for_net:.5f}, SSIM: {avg_ssim_for_net:.6f}')

        diff_loss=avg_loss_for_net-avg_loss_for_mean
        diff_psnr=avg_psnr_for_net-avg_psnr_for_mean
        diff_ssim=avg_ssim_for_net-avg_ssim_for_mean

        return diff_loss,diff_psnr,diff_ssim

    def validation(self, dataset, epoch):
        """
        apply eval on val dataset
        :param epoch: to save with curent epoch#
        :return: None, but creates the files in output folder
        """
        val_on_entire_validationSet = True
        if val_on_entire_validationSet:
            losses=[]
            psnrs=[]
            ssims=[]

            if not self.config['preload_val_data']:  # needs to load the validation data
                for dataset_idx, (input_father_folder_single, input_father_in_name_single, gt_father_folder_single,
                     gt_father_in_name_single) in enumerate(zip(self.config['val_input_father_dir'],
                                                      self.config['training_input_in_name'],
                                                      self.config['val_gt_father_dir'],
                                                      self.config['training_gt_in_name'])):
                    single_dataset_losses = []
                    single_dataset_psnrs = []
                    single_dataset_ssims = []
                    single_inputs, single_gts = dataset.preload_training_data_single(input_father_folder_single,
                                                                                  input_father_in_name_single,
                                                                                  gt_father_folder_single,
                                                                                  gt_father_in_name_single, verbose_=False)
                    for val_vid_idx, (val_vid_in, val_vid_gt) in enumerate(zip(single_inputs, single_gts)):
                        val_vid_prediction = self.eval_entire_vid(val_vid_in)

                        val_loss, val_psnr, val_ssim = self.calc_loss(val_vid_prediction.transpose(dim0=0,dim1=1), val_vid_gt.transpose(dim0=0,dim1=1), calc_psnrssim=True)
                        val_loss=val_loss.item()
                        single_dataset_losses.append(val_loss); single_dataset_psnrs.append(val_psnr); single_dataset_ssims.append(val_ssim)
                        if not self.config['debug']:
                            val_save_path = os.path.join(self.config['save_dir'], 'validation',f'val_dataset_{dataset_idx}_vid_{val_vid_idx:02d}_frame_0',f'e_{epoch:05d}.png',)
                            utils.write_im_vid(val_save_path, val_vid_prediction[:,0,:,:].unsqueeze(0))
                    single_avg_val_loss = sum(single_dataset_losses) / len(single_dataset_losses)
                    single_avg_val_psnr = sum(single_dataset_psnrs) / len(single_dataset_psnrs)
                    single_avg_val_ssim = sum(single_dataset_ssims) / len(single_dataset_ssims)
                    dataset_name=self.config['datasets'][dataset_idx] if self.config['datasets'] is not None else input_father_folder_single
                    print(f'\t\tVALIDATION AFTER epoch:{epoch}, dataset {dataset_name}, loss:{single_avg_val_loss:.5f}, PSNR: {single_avg_val_psnr:.2f}, SSIM: {single_avg_val_ssim:.3f}')

                    losses = losses+single_dataset_losses; psnrs = psnrs+single_dataset_psnrs; ssims = ssims+single_dataset_ssims

            else:  # already loaded the val data
                for dataset_idx, (single_dataset_val_inputs, single_dataset_val_gt) in enumerate(zip(dataset.val_inputs, dataset.val_gt)):

                    single_dataset_losses = []
                    single_dataset_psnrs = []
                    single_dataset_ssims = []
                    for val_vid_idx, (val_vid_in, val_vid_gt) in enumerate(zip(single_dataset_val_inputs,single_dataset_val_gt)):
                        val_vid_prediction = self.eval_entire_vid(val_vid_in)
                        val_loss, val_psnr, val_ssim = self.calc_loss(val_vid_prediction.transpose(dim0=0, dim1=1), val_vid_gt.transpose(dim0=0, dim1=1), calc_psnrssim=True)
                        single_dataset_losses.append(val_loss.item()); single_dataset_psnrs.append(val_psnr); single_dataset_ssims.append(val_ssim)
                        if not self.config['debug']:
                            val_save_path = os.path.join(self.config['save_dir'], 'validation',f'val_dataset_{dataset_idx}_vid_{val_vid_idx:02d}_frame_0',f'e_{epoch:05d}.png',)
                            utils.write_im_vid(val_save_path, val_vid_prediction[:,0,:,:].unsqueeze(0))

                    single_avg_val_loss = sum(single_dataset_losses) / len(single_dataset_losses)
                    single_avg_val_psnr = sum(single_dataset_psnrs) / len(single_dataset_psnrs)
                    single_avg_val_ssim = sum(single_dataset_ssims) / len(single_dataset_ssims)
                    dataset_name=self.config['datasets'][dataset_idx] if self.config['datasets'] is not None else ''
                    print(f'\t\tVALIDATION AFTER epoch:{epoch}, dataset {dataset_name}, loss:{single_avg_val_loss:.5f}, PSNR: {single_avg_val_psnr:.2f}, SSIM: {single_avg_val_ssim:.3f}')
                    losses = losses+single_dataset_losses; psnrs = psnrs+single_dataset_psnrs; ssims = ssims+single_dataset_ssims

            avg_val_loss=sum(losses)/len(losses) if len(losses)>0 else 0  # can be 0 if internal training
            avg_val_psnr=sum(psnrs)/len(psnrs) if len(psnrs)>0 else 0
            avg_val_ssim=sum(ssims)/len(ssims) if len(ssims)>0 else 0
            if hasattr(self, 'writer'):
                self.writer.add_scalars('val_loss', {'val_loss': avg_val_loss})
                self.writer.add_scalars('avg_val_psnr', {'avg_val_psnr': avg_val_psnr})
                self.writer.add_scalars('avg_val_ssim', {'avg_val_ssim': avg_val_ssim})

            print(f'\t\tVALIDATION AFTER epoch:{epoch}, average over all datasets: loss:{avg_val_loss:.5f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.3f}')
        else: assert False, f'not written yet'

        return avg_val_loss, avg_val_psnr, avg_val_ssim


    def eval_entire_vid(self, eval_vid_in, debug_save_mean_path=None):

        eval_vid_in=torch.transpose(eval_vid_in,dim0=0,dim1=1)  #places frames in Batch dimension
        eval_prediction=self.eval(eval_vid_in, debug_save_mean_path)
        return torch.transpose(eval_prediction, dim0=0, dim1=1)  # switch order back to (C,T,H,W)



    def save_final_val_results(self, dataset):
        """
        apply eval on val dataset
        :param epoch: to save with curent epoch#
        :return: None, but creates the files in output folder
        """
        val_on_entire_validationSet = True
        if not self.config['preload_val_data']:  # needs to load the validation first
            val_inputs, val_gt = dataset.preload_training_data(
                self.config['val_input_father_dir'], self.config['training_input_in_name'],
                self.config['val_gt_father_dir'], self.config['training_gt_in_name'])
        else:
            val_inputs, val_gt = dataset.val_inputs, dataset.val_gt  # use the loaded



        if val_on_entire_validationSet:
            losses=[]
            psnrs=[]
            ssims=[]

            for dataset_idx, (single_dataset_val_inputs, single_dataset_val_gt) in enumerate(zip(val_inputs, val_gt)):
                dataset_name = self.config['datasets'][dataset_idx] if self.config['datasets'] is not None else ''
                single_losses = []; single_psnrs = []; single_ssims = []
                for val_vid_idx, (val_vid_in, val_vid_gt) in enumerate(zip(single_dataset_val_inputs, single_dataset_val_gt)):
                    val_vid_prediction = self.eval_entire_vid(val_vid_in)

                    val_loss, val_psnr, val_ssim = self.calc_loss(val_vid_prediction.transpose(dim0=0, dim1=1), val_vid_gt.transpose(dim0=0, dim1=1), calc_psnrssim=True)

                    single_losses.append(val_loss.item()); single_psnrs.append(val_psnr); single_ssims.append(val_ssim)
                    if not self.config['debug']:
                        val_save_path = os.path.join(self.config['save_dir'], 'validation', f'val_dataset_{dataset_name}_vid_{val_vid_idx:02d}_final_full')
                        utils.write_im_vid(val_save_path, val_vid_prediction)

                single_avg_val_loss = sum(single_losses) / len(single_losses); single_avg_val_psnr = sum(single_psnrs) / len(single_psnrs); single_avg_val_ssim = sum(single_ssims) / len(single_ssims)


                print(
                    f'\t\tFINAL VALIDATION, dataset {dataset_name}: loss:{single_avg_val_loss:.5f}, PSNR: {single_avg_val_psnr:.2f}, SSIM: {single_avg_val_ssim:.3f}')
                losses = losses + single_losses; psnrs = psnrs + single_psnrs; ssims = ssims + single_ssims

            avg_val_loss=sum(losses)/len(losses) if len(losses)>0 else 0
            avg_val_psnr=sum(psnrs)/len(psnrs) if len(psnrs)>0 else 0
            avg_val_ssim=sum(ssims)/len(ssims) if len(ssims)>0 else 0

            print(f'\t\tFINAL VALIDATION, loss:{avg_val_loss:.5f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.3f}')
        else: assert False, f'not written yet'

        return avg_val_loss, avg_val_psnr, avg_val_ssim


    def eval(self, input, debug_save_mean_path=None):
        """
        take the input video and upscale it
        :param data: data_handler object, contains the whole video, on which we run the network to produce an upsampled video
        :return:
        """
        self.net.eval()
        with torch.no_grad():
            if self.config['data_augmenter']['eval_ensemble']:
                ensemble_rots=(0,2) if self.config['data_augmenter']['eval_ensemble_not_mixing_xy'] else (0,1,2,3)
                #eval on all combinations of rotation/flips, and median over the results.
                net_output = self.forward(input.to(self.device), debug_save_mean_path=debug_save_mean_path).unsqueeze(0)
                for rot_k in ensemble_rots:
                    if rot_k!=0:  # skip if non-augmented, already in net_output
                        output_auged = self.forward(input.rot90(k=rot_k, dims=[2, 3]).to(self.device),
                                                    debug_save_mean_path=None).rot90(k=4 - rot_k, dims=[2, 3]).unsqueeze(0)
                        net_output=torch.cat((net_output,output_auged),dim=0)
                    output_auged = self.forward(input.rot90(k=rot_k, dims=[2, 3]).flip(dims=[3]).to(self.device),
                                                debug_save_mean_path=None).flip(dims=[3]).rot90(k=4 - rot_k, dims=[2, 3]).unsqueeze(0)
                    net_output=torch.cat((net_output,output_auged),dim=0)
                if self.config['data_augmenter']['ensemble_method']=="median":
                    net_output=torch.median(net_output,dim=0)[0]
                elif self.config['data_augmenter']['ensemble_method']=="mean":
                    net_output = torch.mean(net_output, dim=0)
                else: assert False

            else:
                net_output = self.forward(input.to(self.device), debug_save_mean_path=debug_save_mean_path)
        self.net.train()

        return net_output



    def save_model(self, epoch=None, overwrite=False, fine_tune=False):
        """
        Saves the model (state-dict, optimizer and lr_sched
        :return:
        """
        folder = os.path.join(self.config['save_dir'], 'saved_models')
        os.makedirs(folder, exist_ok=True)
        if overwrite:
            checkpoint_list = [i for i in os.listdir(folder) if i.endswith('.pth.tar')]
            if len(checkpoint_list) != 0:
                os.remove(os.path.join(folder, checkpoint_list[-1]))
        if not fine_tune:
            filename = 'checkpoint{}.pth.tar'.format('' if epoch is None else '-e{:05d}'.format(epoch))
        else:
            filename = 'checkpoint_fine_tune{}.pth.tar'.format('' if epoch is None else '-e{:05d}'.format(epoch))
        torch.save({'epoch': epoch,
                    'sd': self.net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   os.path.join(folder, filename))

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'], strict=False)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['opt'])
        self.epoch = checkpoint['epoch']
