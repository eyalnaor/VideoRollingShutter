import os
import shlex
from subprocess import check_output
import torch
import PIL
import numpy as np
import math
from matplotlib import pyplot as plt
import resize_right_right.resize_right_right as resize
import subprocess
from shutil import copyfile
import imageio


def send_command(command):
    print(command)
    byte_output = check_output(shlex.split(command))
    out = byte_output.decode("utf-8")
    return out

def plt_imshow_torch_reversed(torch_image,vmin=None,vmax=None, normalize=False, colorbar=False):
    # imshows image from torch [BCTHW]
    # Useful for debugging
    if len(torch_image.shape) == 3:  # grayscale
        np_image = torch_image.cpu().detach().numpy().squeeze(0)
    else:
        np_image=torch_image.cpu().detach().numpy().squeeze(0).transpose((1,2,0))
    if normalize:
        np_image = (np_image-np.min(np_image))/(np.max(np_image)-np.min(np_image))
    plt.figure()
    plt.imshow(np_image,vmin=vmin,vmax=vmax)
    if colorbar:
        plt.colorbar()
    plt.show()

def extract_frames_from_video(video_path, frame_dir, frame_file_glob=r"%05d.png", res=None, makedir=True, verbose=True):
    if makedir and not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)
    if res is None:
        res="-1:-1"
    elif type(res) in (list,tuple):  # reverse order, given 2
        res = f'{res[1]}:{res[0]}'
    elif type(res) == int:  # assume given height, preserve aspect ratio
        res = f'-1:{res[0]}'
    command = """ffmpeg -i %s -start_number 00000 -vf mpdecimate,setpts=N/FRAME_RATE/TB,scale=%s %s""" % (
        video_path,
        res,
        frame_dir + "/" + frame_file_glob,
    )
    print("extracting frames from video")
    send_command(command)

    if verbose:
        print(f'extracted {len(os.listdir(frame_dir))} frames')

def download_youtube_video(url, resolution='bestvideo', start="00:00:00.00", num_seconds="00.00", video_folder=None, video_name=None, save_frames=True, save_txt=True):
    """ from: https://unix.stackexchange.com/a/282413/403616 """
    assert video_folder is not None
    os.makedirs(video_folder,exist_ok=True)
    if video_name is None:
        video_name=os.path.split(video_folder)[-1]
    video_path=os.path.join(video_folder,f'{video_name}.mp4')
    frames_path=video_folder
    txt_path=os.path.join(video_folder, f'youtube_details.txt')

    resolution_dict={'bestvideo': 'bestvideo', '720p': 136, '480p': 135, '360p': 134, '240p': 133,'144p': 140}
    resolution_flag=resolution_dict[resolution]

    # get full url of the video
    command = """youtube-dl -f %s --youtube-skip-dash-manifest -g %s""" % (
        resolution_flag,
        url
    )
    out = send_command(command)
    video_url = out.split('\n')[0]

    # download the relevant part using ffmpeg
    command = """ffmpeg -ss %s -i "%s" -t %s -c copy -y %s""" % (
        start,
        video_url,
        num_seconds,
        video_path,
    )
    send_command(command)
    print('saved video to path:', video_path)
    if save_frames:
        extract_frames_from_video(video_path, frames_path)
    if save_txt:
        f = open(txt_path, "w+")
        f.write(f'youtube url: {url}\r\n')
        f.write(f'resolution: {resolution}\r\n')
        f.write(f'start: {start}\r\n')
        f.write(f'length: {num_seconds}\r\n')
        f.close()

def send_DAIN_command(env_py_path, input_dir, output_dir, input_in_name=None, num_interpolated_frames=480,
                      reverse_order=False, rotation=0, flip='none', save_mp4=True, save_frames=False,
                      verbose=False):
    """
    Helper function to send DAIN command with an external environment.
    Enables easier less-dependant environment setup instead of a unified one with clashing dependencies:
    One for DAIN (or other plug-and-play temporal interpolation method)
    One for our MergeNet+optimization step
    """
    DAIN_path=os.path.join(os.path.dirname(__file__),'DAIN','DAIN_for_VideoRollingShutter.py')
    command=f"{env_py_path} {DAIN_path} -i {input_dir} -o {output_dir} -in {input_in_name} -n {num_interpolated_frames} -re {str(reverse_order)} -ro {rotation} -fl {flip} -sm {str(save_mp4)} -sf {save_frames} -v {str(verbose)}"
    send_command(command)


def load_video_to_torch(video_path,use_torchvision=False):
    """
    Returns video in torch in [1,c,t,h,w]
    """
    if use_torchvision:
        import torchvision
        vid, _, _ = torchvision.io.read_video(video_path)
        vid=vid.type(torch.float32)/255.0
        vid = vid.permute(3,0,1,2).unsqueeze(0).detach().contiguous().cuda()  # switch to BCTHW
        return vid
    else:
        vid = imageio.get_reader(video_path, 'ffmpeg')
        vid_tensor = None
        for i, image in enumerate(vid):
            if vid_tensor is None:
                vid_tensor = torch.tensor(image/255.0).unsqueeze(0).detach().cuda()
            else:
                vid_tensor = torch.cat((vid_tensor,torch.tensor(image/255.0).unsqueeze(0).detach().cuda()),dim=0)
        vid_tensor = vid_tensor.unsqueeze(0).contiguous().permute(0, 4, 1, 2, 3).detach().cuda()

        return vid_tensor.type(torch.float32)

def load_video_folder_to_torch(frames_folder, print_=True, must_include_in_name=None):
    """
    loads to torch from video with image frames. Shape: [1,c,t,h,w]
    """
    frames_paths = sorted([os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.png')])
    if must_include_in_name is not None:
        frames_paths = [frame_path for frame_path in frames_paths if must_include_in_name in os.path.split(frame_path)[-1]]
    if len(frames_paths)==0:
        return None
    torch_video = load_video_from_frame_paths_to_torch(frames_paths,print_=print_)
    return torch_video

def load_video_from_frame_paths_to_torch(frames_paths,print_=True):
    """
    loads to torch from list of image frames. Shape: [1,c,t,h,w]
    frame_paths: list of frame paths (useful for loading subset of frames in folder for instance)
    assumes frames are same size
    """
    example_frame = load_image_to_torch(frames_paths[0])
    video_shape=list(example_frame.shape)
    video_shape.insert(2, len(frames_paths))
    torch_video=torch.zeros(tuple(video_shape),dtype=example_frame.dtype).cuda()
    for idx, frame_path in enumerate(frames_paths):
        if print_ and not idx%100:
            print(f'in load_video_folder_to_torch, idx: {idx}')
        torch_frame = load_image_to_torch(frame_path)
        torch_video[:,:,idx,:,:] = torch_frame
    return torch_video

def load_image_to_torch(image_path):
    gt = np.expand_dims(imread(image_path), 0)
    gt = torch.tensor(gt).contiguous().permute(0, 3, 1, 2).detach().cuda()
    return gt

## Image
def imread(fname, bounds=(0.0, 1.0), mode='RGB', **kwargs):
    image = PIL.Image.open(fname, **kwargs).convert(mode=mode)
    image = _img_to_float32(image, bounds)
    return image

def write_im_vid(path, im_vid, bounds=(0.0, 1.0), **kwargs):
    im_vid=im_vid.squeeze(0)
    if len(im_vid.shape)==3:  # image
        assert path.endswith('png')
        imwrite(path, im_vid, bounds=bounds, **kwargs)
    elif len(im_vid.shape)==4:  # video
        assert not path.endswith('png')
        vidwrite(path, im_vid, bounds=bounds, **kwargs)
    else:
        assert False, f'saving unknown size: {im_vid.shape}'


def imwrite(fname, image, bounds=(0.0, 1.0), **kwargs):
    fname = _check_path(fname)
    image = _img_to_uint8(image, bounds)
    if image.shape[2]==1:  # need to go from gray to rgb
        image=np.tile(image,(1,1,3))
    image = PIL.Image.fromarray(image)
    image.save(fname, **kwargs)

def vidwrite(dirname, video, bounds=(0.0, 1.0), **kwargs):
    #video is in [c,t,h,w]. saves all frames separately
    os.makedirs(dirname,exist_ok=True)
    for i in range(video.shape[1]):
        image=video[:,i,:,:]
        fname = os.path.join(dirname,f'{i:05d}.png')
        imwrite(fname, image, bounds=bounds, **kwargs)
    fps = kwargs['fps'] if 'fps' in kwargs.keys() else None
    if fps is not None:
        cur_dir=os.getcwd()
        os.chdir(dirname)
        subprocess.call(['ffmpeg', '-framerate', str(fps), '-i', '%05d.png', 'output.avi'])
        os.chdir(cur_dir)

def _check_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

def _img_to_float32(image, bounds):
    vmin, vmax = bounds
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.clip((vmax - vmin) * image + vmin, vmin, vmax)
    return image

def _img_to_uint8(image, bounds):
    if isinstance(image, torch.Tensor):
        image = _torch_to_np(image)

    if image.dtype != np.uint8:
        vmin, vmax = bounds
        image = (image.astype(np.float32) - vmin) / (vmax - vmin)
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image

def _torch_to_np(image):
    image = to_numpy(image)
    if image.ndim == 3:
        image = image.transpose((1, 2, 0))
    elif image.ndim == 4:
        image = image.transpose((0, 2, 3, 1))
    else:
        raise ValueError()
    return image

def to_numpy(tensor, clone=True):
    tensor = tensor.detach()
    tensor = tensor.clone() if clone else tensor
    return tensor.cpu().numpy()

def sample_frames_in_folder(frames_folder,sample_every=None,target_folder=None, start_frame=0, num_frames=None, rename=False, rename_restart_count=False, rename_suffix=''):
    """
    takes a folder with frames, and samples every N frames. Useful for instance to match speed of GS video (high fps) to generated RS
    """
    frames = sorted([frame for frame in os.listdir(frames_folder) if frame.endswith('.png')])
    if target_folder is None:
        target_folder = os.path.join(frames_folder,f'sampled_every_{sample_every}')
    os.makedirs(target_folder, exist_ok=True)
    sampled_idx=0
    for frame_idx in range(start_frame,len(frames), sample_every):
        if num_frames is not None and sampled_idx >= num_frames:
            return  #cut off
        frame_name=frames[frame_idx]
        orig_frame_path=os.path.join(frames_folder,frame_name)
        if rename:
            frame_naming_num = sampled_idx if rename_restart_count else frame_idx
            frame_name_final = f'{frame_naming_num:05d}{rename_suffix}.png'
        else:
            frame_name_final=frame_name
        new_frame_path = os.path.join(target_folder, frame_name_final)
        copyfile(orig_frame_path,new_frame_path)
        sampled_idx+=1

def crop_all_frames_in_folder(source_folder, cropped_folder=None, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, rename=False):
    if cropped_folder is None:
        cropped_folder = source_folder  # replaces in place
    os.makedirs(cropped_folder, exist_ok=True)

    orig_frames = sorted([frame for frame in os.listdir(source_folder) if frame.endswith('.png')])
    orig_size = load_image_to_torch(os.path.join(source_folder, orig_frames[0])).shape
    for f_idx, frame_name in enumerate(orig_frames):
        orig_im = load_image_to_torch(os.path.join(source_folder, frame_name))
        cropped_im=orig_im[:,:,crop_top:orig_size[2]-crop_bottom,crop_left:orig_size[3]-crop_right]
        write_im_vid(os.path.join(cropped_folder, f'{f_idx:05d}.png' if rename else frame_name), cropped_im)

def get_num_rows(frames_folder, in_name=None):
    """
    load example frame from folder, returns num of rows
    """
    frames=[f for f in os.listdir(frames_folder) if f.endswith('.png')]
    if in_name is not None:
        frames = [f for f in frames if in_name in f]
    return load_image_to_torch(os.path.join(frames_folder,frames[0])).shape[2]









############################################################################
# metrics
############################################################################

def PSNR_SSIM_video_avg_frames(vid1, vid2, mask=None):
    #simply returns together..
    return PSNR_video_avg_frames(vid1, vid2, mask=mask), SSIM_video_avg_frames(vid1, vid2)


def PSNR_video_avg_frames(vid1, vid2, mask=None):
    # calculates PSNR for video as avg of its frames' PSNR
    assert vid1.shape==vid2.shape
    PSNRs=[]
    if mask is None:
        for idx in range(vid1.shape[2]):
            PSNRs.append(PSNR(vid1[:, :, idx, :, :].cuda(), vid2[:, :, idx, :, :].cuda()))
    else:
        for idx in range(vid1.shape[2]):
            PSNRs.append(PSNR(vid1[:, :, idx, :, :].cuda(), vid2[:, :, idx, :, :].cuda(),mask[:, 0:1, idx, :, :].cuda()))
    return sum(PSNRs)/len(PSNRs)


def PSNR(img1, img2, mask=None):
    if mask is not None:
        mask = mask.cuda()
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


############################################################################
# Rolling-Shutter util functions
############################################################################

def synthesize_rolling_shutter(video_path, target_folder, RS_direction='ver', RS_resolution=None,
                               GS_frames_between_RS_frames=None, RS_fps=None):
    """
    Synthesizes a RS video from a high speed video
    :param video_path: location of high speed video. video or folder with frames
    :param RS_direction: 'ver' or 'hor'
    :param RS_resolution: If want lower res for each frame. Useful for "less" RS effect by smaller res
    :param GS_frames_between_RS_frames: Can be more/less than resolution and cause duty_cycle<1 or temporal overlap respectively
    :param target_folder: saves RS frames and video there
    """

    # Load video to tensor
    input_is_video = video_path.endswith(('.mp4','.MP4'))
    if input_is_video:
        GS_video = load_video_to_torch(video_path)
    else:
        GS_video = load_video_folder_to_torch(video_path)

    # Calc RS video size
    if RS_resolution is None:
        RS_resolution = GS_video.shape[3:]
    if GS_frames_between_RS_frames is None:
        GS_frames_between_RS_frames = RS_resolution[0] if RS_direction == 'ver' else RS_resolution[1]
    GS_frame_num = GS_video.shape[2]

    res_RS_direction = RS_resolution[0] if RS_direction == 'ver' else RS_resolution[1]
    RS_frame_num = (GS_frame_num - res_RS_direction) // GS_frames_between_RS_frames + 1  # how many RS frames can be created
    if RS_frame_num<1:
        print(f'not enough frames in {video_path}, RS direction {RS_direction} with res {RS_resolution}. returning')
        return
    RS_video_shape = (1, GS_video.shape[1], RS_frame_num, *RS_resolution)
    GS_video_shape = (1, GS_video.shape[1], GS_video.shape[2], *RS_resolution)

    # Resize GS video if needed
    GS_resolution = GS_video.shape[3:]
    if not RS_resolution[0] == GS_resolution[0] or not RS_resolution[1] == GS_resolution[1]:
        GS_video = resize.resize(GS_video, out_shape=GS_video_shape, interp_method=resize.interp_methods.cubic)
    # Synthesize RS tensor frame by frame
    RS_video = torch.zeros(RS_video_shape, dtype=GS_video.dtype)
    for RS_frame_idx in range(RS_frame_num):
        RS_frame = torch.zeros_like(RS_video[:, :, RS_frame_idx, :, :])
        if RS_direction == 'ver':
            for row in range(RS_resolution[0]):
                wanted_GS_frame = GS_frames_between_RS_frames * RS_frame_idx + row
                RS_frame[:, :, row, :] = GS_video[:, :, wanted_GS_frame, row, :]
        else:
            for col in range(RS_resolution[1]):
                wanted_GS_frame = GS_frames_between_RS_frames * RS_frame_idx + col
                RS_frame[:, :, :, col] = GS_video[:, :, wanted_GS_frame, :, col]
        RS_video[:, :, RS_frame_idx, :, :] = RS_frame

    # Save RS frames & video
    write_im_vid(target_folder, RS_video.squeeze(0), kwargs={'fps': RS_fps})


def GS_from_dense_RS_folder_vids(dense_RS_folder, GS_output_folder=None, RS_direction='ver', GS_frames_to_make=None, GS_centered_around='left', fastec_take_two_from_each_RS=False):
    """
    assumes for each frame has first (orig) frame as 'XXXXX_00000.png', and vid of 'XXXXX_DAIN.mp4' with rows-1 frames, in fps 30
    GS_frames_to_make: list, can be of one frame
    GS_centered_around: 'left', 'center', 'right', 'all'. left: top row is aligned
    """

    RS_frames = sorted([frame for frame in os.listdir(dense_RS_folder) if frame.endswith('.png')])
    RS_vids = sorted([vid for vid in os.listdir(dense_RS_folder) if vid.endswith(('.mp4','.MP4')) and '_DAIN' in vid])
    frame_shape = load_image_to_torch(os.path.join(dense_RS_folder,RS_frames[0])).shape
    orig_frames = sorted(list(set([int(i.split('_')[0]) for i in RS_frames])))
    if GS_frames_to_make is None:
        if GS_centered_around=='left':
            GS_frames_to_make=orig_frames[1:]  # can't do leftmost
        elif GS_centered_around == 'center':
            GS_frames_to_make = orig_frames[1:-1]  # can't do leftmost of rightmost
        elif GS_centered_around == 'right':
            GS_frames_to_make = orig_frames[:-1]  # can't do rightmost
        elif GS_centered_around == 'all':
            assert False, f'not implemented yet..'
        else:
            assert False

    if GS_output_folder is None:
        GS_output_folder=os.path.join(dense_RS_folder,f'GS_made')
    os.makedirs(GS_output_folder,exist_ok=True)
    for frame_idx in GS_frames_to_make:
        frame_numbering = frame_idx
        GS_path = os.path.join(GS_output_folder, f'GS_{frame_numbering:05d}_{GS_centered_around}.png')
        if os.path.isfile(GS_path):  # skip if already made
            print(f'GS frame already exists, skipping: {GS_path}')
            continue
        if GS_centered_around == 'left':  # needs all of idx-1 interpolated, and idx orig, to have top row aligned with idx's top row
            vid_name = f'{frame_idx-1:05d}_DAIN.mp4'
            last_frame_name = f'{frame_idx:05d}_00000.png'
            assert vid_name in RS_vids and last_frame_name in RS_frames
            vids = [os.path.join(dense_RS_folder,vid_name)]
            last_frame = os.path.join(dense_RS_folder, last_frame_name)
            frames=[last_frame]
        elif GS_centered_around == 'center':  # needs the two videos, and frame in between
            vid_before_name = f'{frame_idx - 1:05d}_DAIN.mp4'
            vid_after_name = f'{frame_idx:05d}_DAIN.mp4'
            assert vid_before_name in RS_vids and vid_after_name in RS_vids
            vids = [os.path.join(dense_RS_folder, vid_before_name), os.path.join(dense_RS_folder, vid_after_name)]

            middle_frame_name = f'{frame_idx:05d}_00000.png'
            assert middle_frame_name in RS_frames
            middle_frame = os.path.join(dense_RS_folder, middle_frame_name)
            frames = [middle_frame]

        else:
            assert False, f'not implemented yet...'
        print(f'starting generating GS frame #{frame_idx} in folder: {dense_RS_folder}. In total: {GS_frames_to_make}')
        GS_frame = GS_from_dense_RS_single_vid(frames, vids, GS_centered_around, RS_direction, fastec_take_two_from_each_RS=fastec_take_two_from_each_RS)

        write_im_vid(GS_path,GS_frame.squeeze())

def GS_from_dense_RS_single_vid(frames, vids, GS_centered_around='left', RS_direction='ver', fastec_take_two_from_each_RS=False):
    """
    frames - list of needed frames, according to GS_centered_around
    vids - vids with interpolated frames
    """
    frame_example = load_image_to_torch(frames[0])
    frame_shape=frame_example.shape

    GS_frame=torch.zeros_like(frame_example)
    RS_needed = frame_shape[2] if RS_direction=='ver' else frame_shape[3]
    RS_vid_dimension = 3 if RS_direction=='ver' else 4
    if GS_centered_around == 'left':  # given frame is the TOP row, will be last
        assert len(vids) == 1
        vid = vids[0]
        vid_loaded=load_video_to_torch(vid)
        if fastec_take_two_from_each_RS:  # since fastec's synthetic RS dataset was built with 2 frames from each GS frame, added option to "reverse" in same manner
            assert vid_loaded.shape[2] == RS_needed//2 - 1 and len(frames) == 1  # 1 for the original additional frame, gives 2 rows
            if RS_direction == 'ver':
                assert vid_loaded.shape[2]==vid_loaded.shape[RS_vid_dimension]//2-1  # assert right number of frames in the video
                for vid_frame_idx in range(vid_loaded.shape[2]):  # for each frame in the vid
                    GS_idx = RS_needed - 2 * vid_frame_idx - 1
                    GS_frame[:,:,GS_idx,:] = vid_loaded[:, :, vid_frame_idx, GS_idx, :] #bottom row of the 2
                    GS_frame[:,:,GS_idx-1,:] = vid_loaded[:, :, vid_frame_idx, GS_idx-1, :] #top row of the 2
                # now final frame - in top 2 rows
                frame_loaded = load_image_to_torch(frames[0])
                GS_frame[:, :, 1, :] = frame_loaded[:,:,1,:]
                GS_frame[:, :, 0, :] = frame_loaded[:,:,0,:]
            else: assert False, f'not written yet'

        else:
            assert vid_loaded.shape[2] == RS_needed - 1  and len(frames) == 1  # 1 for the original additional frame
            #since last row comes from first frame, needs to reverse
            if RS_direction == 'ver':
                for vid_frame_idx in range(vid_loaded.shape[RS_vid_dimension]-1):
                    GS_idx = RS_needed - vid_frame_idx - 1
                    GS_frame[:,:,GS_idx,:] = vid_loaded[:, :, vid_frame_idx, GS_idx, :]
                # now final frame - in top row
                frame_loaded = load_image_to_torch(frames[0])
                GS_frame[:, :, 0, :] = frame_loaded[:,:,0,:]
            else:
                assert False, f'not written yet'
    elif GS_centered_around == 'center':
        CENTER=frame_shape[2]//2
        print(f'using frame_shape[2]//2 as center: {CENTER}')

        assert len(vids) == 2 and len(frames) == 1
        vid_before = load_video_to_torch(vids[0])
        vid_after = load_video_to_torch(vids[1])
        middle_frame = load_image_to_torch(frames[0])

        if fastec_take_two_from_each_RS:
            if RS_direction == 'ver':
                assert vid_after.shape[2] == vid_before.shape[2] == RS_needed // 2 - 1 and len(frames) == 1  # 1 for the original additional frame, gives 2 rows
                #place rows from vid after (top rows)
                for vid_frame_idx_after in range(CENTER//2):  # for each frame in the first half of the after vid - gives 2 rows to the frame
                    GS_idx = (CENTER//2)*2 - 2 * vid_frame_idx_after - 1
                    GS_frame[:,:,GS_idx,:] = vid_after[:, :, vid_frame_idx_after, GS_idx, :] #bottom row of the 2
                    GS_frame[:,:,GS_idx-1,:] = vid_after[:, :, vid_frame_idx_after, GS_idx-1, :] #top row of the 2
                assert GS_idx-1==0  # last row placed: top row
                #place rows from center frame
                frame_loaded = load_image_to_torch(frames[0])
                GS_frame[:, :, (CENTER//2)*2+1, :] = frame_loaded[:, :, (CENTER//2)*2+1, :]
                GS_frame[:, :, (CENTER//2)*2, :] = frame_loaded[:, :, (CENTER//2)*2, :]
                #place rows from vid before (bottom rows)
                for vid_frame_idx_before in range(CENTER//2,vid_before.shape[2]): # for each frame in the second half of the after vid - gives 2 rows to the frame
                    GS_idx = RS_needed - 2 * vid_frame_idx_before + (CENTER // 2) * 2 - 2  # from 479 to 242
                    GS_frame[:,:,GS_idx,:] = vid_before[:, :, vid_frame_idx_before, GS_idx, :]
                    GS_frame[:,:,GS_idx+1,:] = vid_before[:, :, vid_frame_idx_before, GS_idx+1, :]
            else: assert False, f'not written yet'
        else:
            if RS_direction == 'ver':
                # from before: bottom rows, but reverse order
                for vid_frame_idx in range(CENTER,frame_shape[2]-1):
                    row_idx = RS_needed - vid_frame_idx + CENTER - 1
                    GS_frame[:,:,row_idx,:] = vid_before[:, :, vid_frame_idx, row_idx, :]
                # from middle frame - in middle
                GS_frame[:, :, CENTER, :] = middle_frame[:, :, CENTER, :]
                # from after: top rows, in reverse
                for vid_frame_idx in range(CENTER):
                    row_idx = CENTER -1 - vid_frame_idx
                    GS_frame[:,:,row_idx,:] = vid_after[:, :, vid_frame_idx, row_idx, :]
            else: assert False, f'not written yet'
    else:
        assert False, f'not written yet'
    return GS_frame
