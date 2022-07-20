import math
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import ndimage
from torch.utils import data
import utils
import Network
import sys
import utils


class merge_DataHandler(data.Dataset):
    def __init__(self, config, device):
        """
        create a DataHandler instance. config has the details needed
        """
        self.config = config
        self.device = device

        if self.config['preload_training_data']:
            self.training_inputs, self.training_gts = self.preload_training_data(
                self.config['training_input_father_dir'], self.config['training_input_in_name'],
                self.config['training_gt_father_dir'], self.config['training_gt_in_name'])
        else:
            self.training_vid_names, self.training_frame_names = self.load_training_data_meta(
                self.config['training_input_father_dir'], self.config['training_input_in_name'])

        if self.config['preload_val_data']:
            self.val_inputs, self.val_gt = self.preload_training_data(
                self.config['val_input_father_dir'], self.config['training_input_in_name'],
                self.config['val_gt_father_dir'], self.config['training_gt_in_name'])  # assuming now train/val have same "in_name"
        else:
            self.val_vid_names, self.val_frame_names = self.load_training_data_meta(self.config['val_input_father_dir'],
                                                                                    self.config['training_input_in_name'])


    def preload_training_data(self, input_father_folder_list, input_father_in_name_list, gt_father_folder_list, gt_father_in_name_list):
        """ loads training data. inputs in format: [augs(16),C,T,H,W], gt: [C,T,H,W]
        returns list of lists (for multiple datasets) with inputs/gts in same order. Needed in list because vids have different lengths/sizes """
        inputs,gts=[],[]
        for (input_father_folder_single, input_father_in_name_single, gt_father_folder_single, gt_father_in_name_single) in zip(input_father_folder_list, input_father_in_name_list, gt_father_folder_list, gt_father_in_name_list):
            single_inputs, single_gts = self.preload_training_data_single(input_father_folder_single, input_father_in_name_single,
                                                                          gt_father_folder_single, gt_father_in_name_single)
            inputs.append(single_inputs)
            gts.append(single_gts)
        return inputs, gts


    def preload_training_data_single(self, input_father_folder, input_father_in_name, gt_father_folder, gt_father_in_name, verbose_=True):
        """ loads training data. inputs in format: [augs(16),C,T,H,W], gt: [C,T,H,W]
        returns list of lists (for multiple datasets) with inputs/gts in same order. Needed in list because vids have different lengths/sizes """

        augmentations=sorted([subdir for subdir in os.listdir(input_father_folder) if os.path.isdir(os.path.join(input_father_folder, subdir))])
        aug_example=os.path.join(input_father_folder,augmentations[0])
        vids=sorted([subdir for subdir in os.listdir(aug_example) if os.path.isdir(os.path.join(aug_example, subdir))])
        to = {'device': self.device, 'dtype': torch.float32}
        training_inputs=[]
        training_gts=[]
        for idx,vid_name in enumerate(vids):
            if verbose_:
                print(f'in preload_training_data, vid #{idx+1} out of {len(vids)}')
            #load aug0 first to get shape instead of reasigning after each aug
            in_GS_made=True if "GS_made" in os.listdir(f'{input_father_folder}/{augmentations[0]}/{vid_name}') else False
            aug0_path = f'{input_father_folder}/{augmentations[0]}/{vid_name}{"/GS_made" if in_GS_made else ""}'
            aug0_vid=utils.load_video_folder_to_torch(aug0_path, print_=False, must_include_in_name=input_father_in_name)
            input_vid=torch.zeros((len(augmentations),*aug0_vid.shape[1:]),**to)
            input_vid[0,...]=aug0_vid
            for aug_idx, aug in enumerate(augmentations[1:]):
                in_GS_made_ = True if "GS_made" in os.listdir(f'{input_father_folder}/{aug}/{vid_name}') else False
                aug_path = f'{input_father_folder}/{aug}/{vid_name}{"/GS_made" if in_GS_made_ else ""}'
                aug_vid = utils.load_video_folder_to_torch(aug_path, print_=False,must_include_in_name=input_father_in_name)
                input_vid[aug_idx+1, ...] = aug_vid  # +1 since 0 is inside already
            input_vid = input_vid.view((-1, *input_vid.shape[2:]))  # merge the augmentations & RGB channels to a single channels dimension
            training_inputs.append(input_vid)
            gt_path=os.path.join(gt_father_folder,vid_name)
            gt_vid=utils.load_video_folder_to_torch(gt_path, print_=False, must_include_in_name=gt_father_in_name)[0,...]  # the [0,...] is to lose the batch dimension
            if input_father_in_name == 'left':
                gt_vid=gt_vid[:,1:,:,:]  # remove first
            elif input_father_in_name == 'center':
                gt_vid=gt_vid[:,1:-1,:,:]  # remove first and last
            else: assert False
            assert gt_vid.shape[1:]==input_vid.shape[1:]
            training_gts.append(gt_vid)
        return training_inputs, training_gts

    def load_training_data_meta(self, input_father_dir_multiple, input_in_name_multiple):
        """ loads training data META, so could be drawn from and loaded during runtime, for memory reasons. inputs in format: [augs(16),C,T,H,W], gt: [C,T,H,W]
        returns lists of lists. external is for all datasets, internal is video names/lengths for the draw.
        Datasets are in same order as config['datasets_ratios'] needed for the draw """

        print(f'in load_training_data_meta, have {len(self.config["training_input_father_dir"])} datasets/alignments')

        vid_names, frame_names = [], []
        for (single_training_input_father_dir, single_training_input_in_name) in zip(input_father_dir_multiple, input_in_name_multiple):
            single_training_vid_names, single_training_frame_names = self.load_training_data_meta_single(single_training_input_father_dir,
                                                                                                         single_training_input_in_name)
            vid_names.append(single_training_vid_names)
            frame_names.append(single_training_frame_names)
        return vid_names, frame_names


    def load_training_data_meta_single(self, input_father_folder, input_father_in_name):
        """ loads training data META, so could be drawn from and loaded during runtime, for memory reasons. inputs in format: [augs(16),C,T,H,W], gt: [C,T,H,W]
        returns lists with inputs/gts in same order. Needed in list because vids have different lengths/sizes """
        augmentations=sorted([subdir for subdir in os.listdir(input_father_folder) if os.path.isdir(os.path.join(input_father_folder, subdir))])
        aug_example=os.path.join(input_father_folder,augmentations[0])
        vid_names=sorted([subdir for subdir in os.listdir(aug_example) if os.path.isdir(os.path.join(aug_example, subdir))])
        to = {'device': self.device, 'dtype': torch.float32}
        # need for each vid to also have the number of frames

        frame_names=[]
        for idx,vid_name in enumerate(vid_names):
            in_GS_made_ = True if "GS_made" in os.listdir(f'{input_father_folder}/{augmentations[0]}/{vid_name}') else False
            aug0_path = f'{input_father_folder}/{augmentations[0]}/{vid_name}{"/GS_made" if in_GS_made_ else ""}'
            cur_frames_names = sorted([frame for frame in os.listdir(aug0_path) if frame.endswith('.png')])
            if input_father_in_name is not None:
                cur_frames_names = [frames_name for frames_name in cur_frames_names if input_father_in_name in frames_name]
            frame_names.append(cur_frames_names)
        return vid_names, frame_names


    def load_frame(self, input_father_folder, input_father_in_name, drawn_vid_name, drawn_frame_name, gt_father_dir, gt_in_name):
        to = {'device': self.device, 'dtype': torch.float32}
        augs = sorted([subdir for subdir in os.listdir(input_father_folder) if os.path.isdir(os.path.join(input_father_folder, subdir))])

        in_GS_made = True if "GS_made" in os.listdir(f'{input_father_folder}/{augs[0]}/{drawn_vid_name}') else False
        aug0_path = f'{input_father_folder}/{augs[0]}/{drawn_vid_name}{"/GS_made" if in_GS_made else ""}/{drawn_frame_name}'
        aug0_frame = utils.load_image_to_torch(aug0_path)
        # assert aug0_frame.dtype == torch.float32
        input_ = torch.zeros((len(augs), *aug0_frame.shape[1:]), **to)
        for aug_idx, aug in enumerate(augs):
            in_GS_made = True if "GS_made" in os.listdir(f'{input_father_folder}/{aug}/{drawn_vid_name}') else False
            aug_path = f'{input_father_folder}/{aug}/{drawn_vid_name}{"/GS_made" if in_GS_made else ""}/{drawn_frame_name}'
            aug_frame = utils.load_image_to_torch(aug_path)
            input_[aug_idx,...] = aug_frame

        input_ = input_.view((-1, *input_.shape[2:]))  # merge the augmentations & RGB channels to a single channels dimension

        # now load gt. Need to translate frame_name to the relevant gt frame_name
        drawn_frame_num=int(drawn_frame_name.split('_')[1])

        if gt_in_name in ['first.png', 'middle.png']:  # fastec
            gt_path = os.path.join(gt_father_dir, drawn_vid_name, f'{drawn_frame_num:03d}_global_{gt_in_name}')
        elif gt_in_name in ['gs_f.png', 'gs_m.png']:  # carla
            gt_path = os.path.join(gt_father_dir, drawn_vid_name, f'{drawn_frame_num:04d}_{gt_in_name}')
        else: assert False, f'unknown dataset. Cumbersome, can be cleaner...'
        gt_ = utils.load_image_to_torch(gt_path)[0,...]  # load and lose the batch dimension

        return input_, gt_



    def __len__(self):
        # return self.video_shape[0]
        return self.config['num_iter_per_epoch']* self.config['batch_size']

    def __getitem__(self, item):
        np.random.seed()

        if self.config['preload_training_data']:  # data already loaded, only extract
            drawn_vid_idx=np.random.randint(len(self.training_inputs))
            input_vid=self.training_inputs[drawn_vid_idx]
            gt_vid=self.training_gts[drawn_vid_idx]
            drawn_frame_idx = np.random.randint(gt_vid.shape[1])
            input_=input_vid[:,drawn_frame_idx,:,:]
            gt_=gt_vid[:,drawn_frame_idx,:,:]

        else:
            datasets_num = len(self.training_vid_names)
            drawn_dataset_idx = np.random.choice(np.arange(datasets_num), p=self.config['datasets_ratios'])
            drawn_vid_idx=np.random.randint(len(self.training_vid_names[drawn_dataset_idx]))
            drawn_vid_name=self.training_vid_names[drawn_dataset_idx][drawn_vid_idx]

            frame_names_in_drawn_vid=self.training_frame_names[drawn_dataset_idx][drawn_vid_idx]
            drawn_frame_idx = np.random.randint(len(frame_names_in_drawn_vid))
            drawn_frame_name = frame_names_in_drawn_vid[drawn_frame_idx]

            gt_in_name = self.config['training_gt_in_name'][drawn_dataset_idx]
            if gt_in_name == 'internal_ystart':  # when internal_ystart - take last part, needed for relevant start row
                gt_in_name=drawn_frame_name.split('_')[-1]
            input_, gt_ = self.load_frame(self.config['training_input_father_dir'][drawn_dataset_idx],
                                          self.config['training_input_in_name'][drawn_dataset_idx], drawn_vid_name,
                                          drawn_frame_name,
                                          self.config['training_gt_father_dir'][drawn_dataset_idx],
                                          gt_in_name)

        #now have image&gt, check if needs augmentations.
        if self.config['data_augmenter']['crop_size'] is not None:
            #crop
            crop_without_edges_percent=self.config['data_augmenter']['crop_without_edge_percent']
            T_low=int(input_.shape[1]*crop_without_edges_percent)
            T_high=int((1-crop_without_edges_percent)*input_.shape[1] - self.config['data_augmenter']['crop_size']+1)
            L_low=int(input_.shape[2]*crop_without_edges_percent)
            L_high=int((1-crop_without_edges_percent)*input_.shape[2] - self.config['data_augmenter']['crop_size']+1)
            TL_corner = (np.random.randint(low=T_low, high=T_high),
                         np.random.randint(low=L_low, high=L_high))

            input_ = input_[:, TL_corner[0]:TL_corner[0] + self.config['data_augmenter']['crop_size'],
                     TL_corner[1]:TL_corner[1] + self.config['data_augmenter']['crop_size']]
            gt_ = gt_[:, TL_corner[0]:TL_corner[0] + self.config['data_augmenter']['crop_size'],
                     TL_corner[1]:TL_corner[1] + self.config['data_augmenter']['crop_size']]
        if self.config['data_augmenter']['flip_hor_prob'] > 0:
            flip_hor=np.random.binomial(1, self.config['data_augmenter']['flip_hor_prob'])
            if flip_hor:
                input_=input_.flip(dims=[2])
                gt_=gt_.flip(dims=[2])
        if self.config['data_augmenter']['flip_ver_prob'] > 0:
            flip_ver=np.random.binomial(1, self.config['data_augmenter']['flip_ver_prob'])
            if flip_ver:
                input_=input_.flip(dims=[1])
                gt_=gt_.flip(dims=[1])
        if self.config['data_augmenter']['rot_prob'] > 0:
            rot_ = np.random.binomial(1, self.config['data_augmenter']['rot_prob'])
            if rot_:
                if self.config['data_augmenter']['eval_ensemble_not_mixing_xy']:
                    rot_k = 2  # 180 deg, not mixing xy
                else:
                    assert input_.shape[1] == input_.shape[2]  # for rotation or 90,270 need image to be square, for the batch
                    rot_k=np.random.randint(1,4)  #1/2/3 (decided to rotate, so not 0)
                input_=input_.rot90(k=rot_k,dims=[1,2])
                gt_=gt_.rot90(k=rot_k,dims=[1,2])

        return input_, gt_

