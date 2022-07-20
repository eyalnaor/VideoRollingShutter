import time
import os
import sys
sys.path.append(os.getcwd())
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy
import networks
from my_args import args
from imageio import imread, imsave
from AverageMeter import *
import shutil
import imageio
import time
from matplotlib import pyplot as plt
from PIL import Image



# torch.backends.cudnn.benchmark = False is requires from some reason
# see: https://github.com/pytorch/pytorch/issues/35870#issuecomment-607867569
torch.backends.cudnn.benchmark = False

DO_MiddleBurryOther = True
input_father_dir = args.input_dir
output_father_dir = args.result_dir
reverse_order = eval(args.reverse_order)
rotation = args.rot
flip = args.flip
save_augmented = eval(args.save_augmented)
print(f'reverse_order: {str(reverse_order)}, rotation: {rotation}, flip: {flip}')

if not os.path.exists(output_father_dir):
    os.makedirs(output_father_dir,exist_ok=True)  # exist_ok since with multiple simultaneous runs from server may be needed


if args.num_interpolated_frames is not None:
    args.time_step = 1/args.num_interpolated_frames

model = networks.__dict__[args.netName](    channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

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
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval()  # deploy mode

use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()

if DO_MiddleBurryOther:
    subdir = [i for i in os.listdir(input_father_dir) if os.path.isdir(os.path.join(input_father_dir, i))]
    add_random=False
    gen_dir = os.path.join(output_father_dir, f'{args.result_suffix}{"_"+unique_id if add_random else ""}')
    os.makedirs(gen_dir, exist_ok=True)

    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()
    for dir in subdir:
        os.makedirs(os.path.join(gen_dir, dir), exist_ok=True)

        #inside subdir, iterate for all relevant pairs:
        subdir_path=os.path.join(input_father_dir, dir)
        frames = sorted([os.path.join(subdir_path, frame) for frame in os.listdir(subdir_path) if frame.endswith('.png')])

        if reverse_order:
            frames.reverse()


        assert args.first_frame_to_do==-1 or len(frames)>=args.first_frame_to_do+args.num_frames_to_do+1, f'assert - not enough frames in {subdir_path} for start: {args.first_frame_to_do}, num: {args.num_frames_to_do}'
        if args.first_frame_to_do == -1:  # do all..
            first_idx=0
            num_pairs=len(frames)-1
        else:
            first_idx=args.first_frame_to_do
            num_pairs=args.num_frames_to_do





        for first_frame_idx in range(first_idx,first_idx+num_pairs):

            ### check if movie exists. If so - skip
            if reverse_order or rotation != 0 or flip != 'none':
                some_augmentation_done_ = True
            else:
                some_augmentation_done_ = False

            if not some_augmentation_done_:
                if some_augmentation_done_:  #some augmentation, save in augmented subfolder first
                    original_folder_ = os.path.join(gen_dir, dir, f'input_augmented_config')
                else:
                    original_folder_ = os.path.join(gen_dir, dir)
                movie_name_ = os.path.join(original_folder_, f'{first_frame_idx:05d}_DAIN.mp4')
            else:
                unaugmented_folder_ = os.path.join(gen_dir, dir)
                if reverse_order:  #reverse first and second, and reverse the video as well
                    after_reverse_first_frame_idx_ = num_pairs - first_frame_idx - 1
                    mov_idx_ = after_reverse_first_frame_idx_
                else:
                    after_reverse_first_frame_idx_ = first_frame_idx
                    mov_idx_ = after_reverse_first_frame_idx_
                movie_name_ = os.path.join(unaugmented_folder_, f'{mov_idx_:05d}_DAIN.mp4')
            if os.path.isfile(movie_name_):
                print(f'skipping because exists: {movie_name_}')
                continue
            ### check if movie exists. If so - skip
            print(f'starting: {movie_name_}')



            arguments_strFirst = frames[first_frame_idx]
            arguments_strSecond = frames[first_frame_idx+1]

            X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
            X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

            if flip is not 'none':
                if 'hor' in flip:
                    X0=torch.flip(X0,[2])
                    X1=torch.flip(X1,[2])
                if 'ver' in flip:
                    X0=torch.flip(X0,[1])
                    X1=torch.flip(X1,[1])
            if rotation != 0:
                assert rotation%90==0
                X0=torch.rot90(X0,k=rotation//90,dims=(1,2))
                X1=torch.rot90(X1,k=rotation//90,dims=(1,2))

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
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

            torch.set_grad_enabled(False)
            X0 = Variable(torch.unsqueeze(X0,0))
            X1 = Variable(torch.unsqueeze(X1,0))
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()

            proc_end = time.time()

            y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))

            y_ = y_s[save_which]

            proc_timer.update(time.time() -proc_end)
            tot_timer.update(time.time() - end)
            end  = time.time()
            print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
            if use_cuda:
                X0 = X0.data.cpu().numpy()
                if not isinstance(y_, list):
                    y_ = y_.data.cpu().numpy()
                else:
                    y_ = [item.data.cpu().numpy() for item in y_]
                offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
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

            X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                      intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
            offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
            filter = [np.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter]  if filter is not None else None
            X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

            timestep = args.time_step
            numFrames = int(1.0 / timestep) - 1
            time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]

            #save results as frames
            count = 0

            X0=np.round(X0).astype(numpy.uint8)
            X1=np.round(X1).astype(numpy.uint8)
            y_=[np.round(item).astype(numpy.uint8) for item in y_]

            if reverse_order or rotation != 0 or flip != 'none':
                some_augmentation_done = True
            else:
                some_augmentation_done = False


            if (save_augmented and some_augmentation_done) or not some_augmentation_done:
                if some_augmentation_done:  #some augmentation, save in augmented subfolder first
                    original_folder = os.path.join(gen_dir, dir, f'input_augmented_config')
                    os.makedirs(original_folder, exist_ok=True)
                else:
                    original_folder = os.path.join(gen_dir, dir)
                print(f'entering movie_save for augmented')
                im = Image.fromarray(X0)
                im.save(os.path.join(original_folder, f'{first_frame_idx:05d}_{0:05d}.png'))

                movie=np.stack(y_,axis=0)
                movie_name = os.path.join(original_folder, f'{first_frame_idx:05d}_DAIN.mp4')
                imageio.mimwrite(movie_name, movie, macro_block_size=1, quality=10, fps=30)
                if first_frame_idx==first_idx+num_pairs-1: # last pair, so copy last frame
                    im = Image.fromarray(X1)
                    im.save(os.path.join(original_folder, f'{first_frame_idx+1:05d}_{0:05d}.png'))


            if some_augmentation_done:  # some augmentation, undo and save in folder
                unaugmented_folder = os.path.join(gen_dir, dir)
                if rotation != 0:
                    assert rotation % 90 == 0
                    X0 = np.rot90(X0, k=4-(rotation // 90), axes=(0, 1))
                    X1 = np.rot90(X1, k=4-(rotation // 90), axes=(0, 1))
                    y_ = [np.rot90(item, k=4-(rotation // 90), axes=(0, 1)) for item in y_]
                if flip is not 'none':
                    if 'hor' in flip:
                        X0 = np.flip(X0, axis=1)  # now np array and hwc
                        X1 = np.flip(X1, axis=1)
                        y_ = [np.flip(item, axis=1) for item in y_]
                    if 'ver' in flip:
                        X0 = np.flip(X0, axis=0)  # now np array and hwc
                        X1 = np.flip(X1, axis=0)
                        y_ = [np.flip(item, axis=0) for item in y_]

                if reverse_order:  #reverse first and second, and reverse the video as well
                    first_im=np.copy(X1)
                    second_image=np.copy(X0)
                    y_.reverse()
                    after_reverse_first_frame_idx=num_pairs-first_frame_idx-1  # not sure this works for subset, but when all frames in folder is ok
                    mov_idx = after_reverse_first_frame_idx
                    after_reverse_second_frame_idx = after_reverse_first_frame_idx + 1
                else:
                    first_im=np.copy(X0)
                    second_image=np.copy(X1)
                    after_reverse_first_frame_idx = first_frame_idx
                    mov_idx = after_reverse_first_frame_idx
                    after_reverse_second_frame_idx = after_reverse_first_frame_idx + 1

                im = Image.fromarray(first_im)
                im.save(os.path.join(unaugmented_folder, f'{after_reverse_first_frame_idx:05d}_{0:05d}.png'))
                movie=np.stack(y_,axis=0)
                movie_name = os.path.join(unaugmented_folder, f'{mov_idx:05d}_DAIN.mp4')
                imageio.mimwrite(movie_name, movie, macro_block_size=1, quality=10, fps=30)
                if reverse_order and first_frame_idx==first_idx: # first pair in reverse, so save both frames
                    im2 = Image.fromarray(second_image)
                    im2.save(os.path.join(unaugmented_folder, f'{after_reverse_second_frame_idx:05d}_{0:05d}.png'))
                if not reverse_order and first_frame_idx==first_idx+num_pairs-1: # last pair in not reverse, so save both frames
                    im2 = Image.fromarray(second_image)
                    im2.save(os.path.join(unaugmented_folder, f'{after_reverse_second_frame_idx:05d}_{0:05d}.png'))





print('Done')


