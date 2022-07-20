"""
edited version of code from DAIN [1], edited to match use cases of VideoRollingShutter.
Changed only input/ouput format, and callability as a function.

[1]   Bao, Wenbo, et al. "Depth-aware video frame interpolation."
      Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
      If you plan on quoting this paper, please quote [1] as well.
"""

import time
import os
import sys
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy

import networks
from imageio import imread, imsave
from AverageMeter import *
import shutil
import imageio
import time
from matplotlib import pyplot as plt
from PIL import Image

# torch.backends.cudnn.benchmark = False is required from some reason
# see: https://github.com/pytorch/pytorch/issues/35870#issuecomment-607867569
torch.backends.cudnn.benchmark = False

def DAIN_wrapper_for_VRS(input_dir,
                         output_dir,
                         netName='DAIN_slowmotion',
                         input_in_name=None,  # png must have in name. Useful for example for taking only RS frames
                         reverse_order=False,  # augmentation
                         rotation=0,  # augmentation
                         flip='none',  # augmentation
                         num_interpolated_frames=480,  # temporal interpolation factor
                         channels=3,
                         filter_size=4,
                         use_cuda=True,
                         save_which=1,
                         dtype=torch.cuda.FloatTensor,
                         first_frame_to_do=-1,
                         num_frames_to_do=1,
                         save_mp4=True,
                         save_frames=False,
                         verbose=False
                         ):
    if (not save_mp4 and not save_frames) and verbose:
        print(f'*****In DAIN. not saving mp4 or frames, so does nothing.*****')
    DO_MiddleBurryOther = True
    if verbose:
        print(f'reverse_order: {str(reverse_order)}, rotation: {rotation}, flip: {flip}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir,
                    exist_ok=True)  # exist_ok since with multiple simultaneous runs from server may be needed

    time_step = 1 / num_interpolated_frames

    model = networks.__dict__[netName](channel=channels,
                                            filter_size=filter_size,
                                            timestep=time_step,
                                            training=False)

    if use_cuda:
        model = model.cuda()

    SAVED_MODEL = os.path.join(os.path.dirname(__file__),'model_weights','best.pth')
    if os.path.exists(SAVED_MODEL):
        if verbose:
            print("The testing model weight is: " + SAVED_MODEL)
        if not use_cuda:
            pretrained_dict = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(SAVED_MODEL)

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        pretrained_dict = []
    else:
        if verbose:
            print("*****************************************************************")
            print("**** We don't load any trained weights **************************")
            print("*****************************************************************")

    model = model.eval()  # deploy mode

    unique_id = str(random.randint(0, 100000))
    if verbose:
        print("The unique id for current testing is: " + str(unique_id))

    interp_error = AverageMeter()

    if DO_MiddleBurryOther:
        gen_dir = output_dir
        # add_random = False
        # gen_dir = os.path.join(output_dir, f'{result_suffix}{"_" + unique_id if add_random else ""}')
        os.makedirs(gen_dir, exist_ok=True)

        tot_timer = AverageMeter()
        proc_timer = AverageMeter()
        end = time.time()

        frames = sorted(
            [os.path.join(input_dir, frame) for frame in os.listdir(input_dir) if frame.endswith('.png')])

        if input_in_name is not None:
            frames = sorted([f for f in frames if input_in_name in os.path.split(f)[-1]])

        if reverse_order:
            frames.reverse()

        assert first_frame_to_do == -1 or len(
            frames) >= first_frame_to_do + num_frames_to_do + 1, f'assertion error - not enough frames in {input_dir} for start: {first_frame_to_do}, num: {num_frames_to_do}'
        if first_frame_to_do == -1:  # do all..
            first_idx = 0
            num_pairs = len(frames) - 1
        else:
            first_idx = first_frame_to_do
            num_pairs = num_frames_to_do

        for first_frame_idx in range(first_idx, first_idx + num_pairs):

            ### check if movie exists. If so - skip
            if reverse_order or rotation != 0 or flip != 'none':
                some_augmentation_done_ = True
            else:
                some_augmentation_done_ = False

            if not some_augmentation_done_:
                if some_augmentation_done_:  # some augmentation, save in augmented subfolder first
                    original_folder_ = os.path.join(gen_dir, f'input_augmented_config')
                else:
                    original_folder_ = os.path.join(gen_dir)
                movie_name_ = os.path.join(original_folder_, f'{first_frame_idx:05d}_DAIN.mp4')
            else:
                unaugmented_folder_ = os.path.join(gen_dir)
                if reverse_order:  # reverse first and second, and reverse the video as well
                    after_reverse_first_frame_idx_ = num_pairs - first_frame_idx - 1
                    mov_idx_ = after_reverse_first_frame_idx_
                else:
                    after_reverse_first_frame_idx_ = first_frame_idx
                    mov_idx_ = after_reverse_first_frame_idx_
                movie_name_ = os.path.join(unaugmented_folder_, f'{mov_idx_:05d}_DAIN.mp4')
            if os.path.isfile(movie_name_):
                if verbose:
                    print(f'skipping because exists: {movie_name_}')
                continue
            ### check if movie exists. If so - skip
            if verbose:
                print(f'starting: {movie_name_}')

            arguments_strFirst = frames[first_frame_idx]
            arguments_strSecond = frames[first_frame_idx + 1]


            X0 = torch.from_numpy(np.transpose(imread(arguments_strFirst), (2, 0, 1)).astype("float32") / 255.0).type(
                dtype)
            X1 = torch.from_numpy(np.transpose(imread(arguments_strSecond), (2, 0, 1)).astype("float32") / 255.0).type(
                dtype)

            if flip is not 'none':
                if 'hor' in flip:
                    X0 = torch.flip(X0, [2])
                    X1 = torch.flip(X1, [2])
                if 'ver' in flip:
                    X0 = torch.flip(X0, [1])
                    X1 = torch.flip(X1, [1])
            if rotation != 0:
                assert rotation % 90 == 0
                X0 = torch.rot90(X0, k=rotation // 90, dims=(1, 2))
                X1 = torch.rot90(X1, k=rotation // 90, dims=(1, 2))

            y_ = torch.FloatTensor()

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)

            if not channel == 3:
                continue

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft = int((intWidth_pad - intWidth) / 2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight = 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            torch.set_grad_enabled(False)
            X0 = Variable(torch.unsqueeze(X0, 0))
            X1 = Variable(torch.unsqueeze(X1, 0))
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()

            proc_end = time.time()

            y_s, offset, filter = model(torch.stack((X0, X1), dim=0))

            y_ = y_s[save_which]

            proc_timer.update(time.time() - proc_end)
            tot_timer.update(time.time() - end)
            end = time.time()
            if verbose:
                print("*****************current image process time \t " + str(
                    time.time() - proc_end) + "s ******************")
            if use_cuda:
                X0 = X0.data.cpu().numpy()
                if not isinstance(y_, list):
                    y_ = y_.data.cpu().numpy()
                else:
                    y_ = [item.data.cpu().numpy() for item in y_]
                offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
                X1 = X1.data.cpu().numpy()
            else:
                X0 = X0.data.numpy()
                if not isinstance(y_, list):
                    y_ = y_.data.numpy()
                else:
                    y_ = [item.data.numpy() for item in y_]
                offset = [offset_i.data.numpy() for offset_i in offset]
                filter = [filter_i.data.numpy() for filter_i in filter]
                X1 = X1.data.numpy()

            X0 = np.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                      intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
            y_ = [np.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                       intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
            offset = [np.transpose(
                offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for offset_i in offset]
            filter = [np.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter] if filter is not None else None
            X1 = np.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                      intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

            timestep = time_step
            numFrames = int(1.0 / timestep) - 1
            time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]

            count = 0  # for frame saving

            # prepare for saving:
            X0 = np.round(X0).astype(numpy.uint8)
            X1 = np.round(X1).astype(numpy.uint8)
            y_ = [np.round(item).astype(numpy.uint8) for item in y_]

            if reverse_order or rotation != 0 or flip != 'none':
                some_augmentation_done = True
            else:
                some_augmentation_done = False

            if not some_augmentation_done:
                original_folder = gen_dir
                if save_mp4:
                    if verbose:
                        print(f'entering movie_save for augmented')
                    im = Image.fromarray(X0)
                    im.save(os.path.join(original_folder, f'{first_frame_idx:05d}_{0:05d}.png'))
                    movie=np.stack(y_,axis=0)
                    movie_name = os.path.join(original_folder, f'{first_frame_idx:05d}_DAIN.mp4')
                    imageio.mimwrite(movie_name, movie, macro_block_size=1, quality=10, fps=30)
                    if first_frame_idx==first_idx+num_pairs-1: # last pair, so copy last frame
                        im = Image.fromarray(X1)
                        im.save(os.path.join(original_folder, f'{first_frame_idx+1:05d}_{0:05d}.png'))
                if save_frames:
                    shutil.copy(arguments_strFirst, os.path.join(gen_dir, f'{first_frame_idx:05d}_{count:05d}.png'))
                    count = count+1
                    for item, time_offset in zip(y_, time_offsets):
                        arguments_strOut = os.path.join(gen_dir, f'{first_frame_idx:05d}_{count:05d}.png')
                        count = count + 1
                        imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
                    if first_frame_idx==first_idx+num_pairs-1: # last pair, so copy last frame
                        shutil.copy(arguments_strSecond, os.path.join(gen_dir, f'{first_frame_idx+1:05d}_{0:05d}.png'))


            if some_augmentation_done:  # some augmentation, undo and save in folder
                unaugmented_folder = os.path.join(gen_dir)
                if rotation != 0:
                    assert rotation % 90 == 0
                    X0 = np.rot90(X0, k=4 - (rotation // 90), axes=(0, 1))
                    X1 = np.rot90(X1, k=4 - (rotation // 90), axes=(0, 1))
                    y_ = [np.rot90(item, k=4 - (rotation // 90), axes=(0, 1)) for item in y_]
                if flip is not 'none':
                    if 'hor' in flip:
                        X0 = np.flip(X0, axis=1)  # now np array and hwc
                        X1 = np.flip(X1, axis=1)
                        y_ = [np.flip(item, axis=1) for item in y_]
                    if 'ver' in flip:
                        X0 = np.flip(X0, axis=0)  # now np array and hwc
                        X1 = np.flip(X1, axis=0)
                        y_ = [np.flip(item, axis=0) for item in y_]

                if reverse_order:  # reverse first and second, and reverse the video as well
                    first_im = np.copy(X1)
                    second_image = np.copy(X0)
                    y_.reverse()
                    after_reverse_first_frame_idx = num_pairs - first_frame_idx - 1
                    mov_idx = after_reverse_first_frame_idx
                    after_reverse_second_frame_idx = after_reverse_first_frame_idx + 1
                else:
                    first_im = np.copy(X0)
                    second_image = np.copy(X1)
                    after_reverse_first_frame_idx = first_frame_idx
                    mov_idx = after_reverse_first_frame_idx
                    after_reverse_second_frame_idx = after_reverse_first_frame_idx + 1

                if save_mp4:
                    im = Image.fromarray(first_im)
                    im.save(os.path.join(unaugmented_folder, f'{after_reverse_first_frame_idx:05d}_{0:05d}.png'))
                    # movie_save_tic=time.time()
                    movie = np.stack(y_, axis=0)
                    movie_name = os.path.join(unaugmented_folder, f'{mov_idx:05d}_DAIN.mp4')
                    imageio.mimwrite(movie_name, movie, macro_block_size=1, quality=10, fps=30)
                    if reverse_order and first_frame_idx == first_idx:  # first pair in reverse, so save both frames
                        im2 = Image.fromarray(second_image)
                        im2.save(os.path.join(unaugmented_folder, f'{after_reverse_second_frame_idx:05d}_{0:05d}.png'))
                    if not reverse_order and first_frame_idx == first_idx + num_pairs - 1:  # last pair in not reverse, so save both frames
                        im2 = Image.fromarray(second_image)
                        im2.save(os.path.join(unaugmented_folder, f'{after_reverse_second_frame_idx:05d}_{0:05d}.png'))

                if save_frames:  #save frames. Use only for not augmented.. Useful for uping framerate (sharp and by small number), not for RS solving
                    shutil.copy(arguments_strFirst, os.path.join(gen_dir, f'{first_frame_idx:05d}_{count:05d}.png'))
                    count  = count+1
                    for item, time_offset in zip(y_, time_offsets):
                        arguments_strOut = os.path.join(gen_dir, f'{first_frame_idx:05d}_{count:05d}.png')
                        count = count + 1
                        imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
                    if first_frame_idx==first_idx+num_pairs-1: # last pair, so copy last frame
                        shutil.copy(arguments_strSecond, os.path.join(gen_dir, f'{first_frame_idx+1:05d}_{0:05d}.png'))


    if verbose:
        print('Done')


def DAIN_VRS_create_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='path to folder with input frames', default='')
    parser.add_argument('-o', '--output_dir', type=str, help='path to folder to save to', default='')
    parser.add_argument('-in', '--input_in_name', type=str, help='', default='None')
    parser.add_argument('-n', '--num_interpolated_frames', type=int, help='interpolation factor. for RS to GS needs to be num of rows', default='480')
    parser.add_argument('-re', '--reverse_order', type=bool, help='Augmentation. Whether to reverse temporal order', default=False)
    parser.add_argument('-ro', '--rotation', type=int, help='Augmentation. Rotation in 90 multiplication', default=0)
    parser.add_argument('-fl', '--flip', type=str, help='Augmentation. Whether to flip and how. options: none, hor, ver', default='none')
    parser.add_argument('-sm', '--save_mp4', type=bool, help='whether to save results as mp4. faster than frames', default=False)
    parser.add_argument('-sf', '--save_frames', type=bool, help='whether to save results as frames', default=True)
    parser.add_argument('-v', '--verbose', type=bool, help='whether to print verbosely', default=False)
    if parser.input_in_name == 'None':
        parser.input_in_name=None
    return parser


if __name__ == "__main__":
    """
    Useful since called from command line to solve env issues.
    Parses args and calls DAIN_wrapper_for_VRS.
    """
    parser = DAIN_VRS_create_parser()
    args = parser.parse_args()
    DAIN_wrapper_for_VRS(input_dir=args.input_dir,
                         output_dir=args.output_dir,
                         input_in_name=args.input_in_name,
                         reverse_order=args.reverse_order,
                         rotation=args.rotation,
                         flip=args.flip,
                         num_interpolated_frames=args.num_interpolated_frames,
                         save_mp4=args.save_mp4,
                         save_frames=args.save_frames,
                         verbose=args.verbose
                         )