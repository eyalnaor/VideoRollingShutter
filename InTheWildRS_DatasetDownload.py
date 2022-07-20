"""
Since the videos in the newly curated In-The-Wild-RS dataset are from youtube, we publish the code used to generate it.
The steps are:
1. Download selected videos from youtube links, at wanted timestamps.
2. Crop video if wanted. We cropped one of the videos in the dataset for it to have less rows,
   hence more generated RS frames.
3. Apply DAIN [1] on the frames in order to get an extremely high framerate of the scene.
    a. This is akin to getting a "continuous" spacetime volume, which enables us to "sample" aligned RS-GS videos
    b. Since the input video in already in slow-motion, DAIN works extremely well, and can be regarded as GT
    c. The dataset contains multiple instances of the same input videos with different DAIN upsampling scales.
        This is to have different degrees of motion and difficulty. See paper for details.
4. "sample" the RS/GS rows from the "continuous" space-time volume.
5.  Cleanup if wanted, to leave only the final RS/GS videos.

[1]   Bao, Wenbo, et al. "Depth-aware video frame interpolation."
      Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
      If you plan on quoting this paper, please quote [1] as well.
"""
import os
import utils

# from DAIN.DAIN_for_VideoRollingShutter import DAIN_wrapper_for_VRS
from shutil import rmtree

DAIN_env_python_path=f'/net/mraid11/export/groups/iranig/.conda/envs/itaian_DAIN_for_RS_publish/bin/python'
download_folder = f'./In-the-wild-RS'
cleanup_after = True  # if True: will delete the "intermediate" folders that are created during the RS/GS synthesis.

videos_to_youtubeDetails = {
    'bird_horizontal':    {'url': 'https://www.youtube.com/watch?v=W5JxLenJWtg', 'start': '00:04:45.44', 'len': '05.72', 'crop': None,      'DAINs': [40]},
    'eagle':              {'url': 'https://www.youtube.com/watch?v=e-5xPpBxihI', 'start': '00:02:02.88', 'len': '15.56', 'crop': None,      'DAINs': [5,10,20]},
    'flip':               {'url': 'https://www.youtube.com/watch?v=4Nogn9sQ_Lg', 'start': '00:01:11.68', 'len': '10.00', 'crop': None,      'DAINs': [20]},
    'helicopter':         {'url': 'https://www.youtube.com/watch?v=4Nogn9sQ_Lg', 'start': '00:00:00.00', 'len': '09.16', 'crop': None,      'DAINs': [20]},
    'owl':                {'url': 'https://www.youtube.com/watch?v=W5JxLenJWtg', 'start': '00:06:25.36', 'len': '04.16', 'crop': None,      'DAINs': [40]},
    'snake_bite':         {'url': 'https://www.youtube.com/watch?v=W5JxLenJWtg', 'start': '00:07:04.60', 'len': '04.12', 'crop': None,      'DAINs': [40]},
    'taekwondo':          {'url': 'https://www.youtube.com/watch?v=jlzVmOUP1is', 'start': '00:07:52.88', 'len': '24.68', 'crop': None,      'DAINs': [20]},
    'wet_dog':            {'url': 'https://www.youtube.com/watch?v=W5JxLenJWtg', 'start': '00:00:47.48', 'len': '16.00', 'crop': None,      'DAINs': [10,20]},
    'cheetah':            {'url': 'https://www.youtube.com/watch?v=XGlrMNvzMGE', 'start': "00:00:07.00", 'len': "05.00", 'crop': [110, 80], 'DAINs': [10, 20]},
}
#cheetahs, wrong:
#https://www.youtube.com/watch?v=2joz_4l5mzo
#https://www.youtube.com/watch?v=e-5xPpBxihI
#https://www.youtube.com/watch?v=BRcySdyyA-A  -  pseudo license
for name, details in videos_to_youtubeDetails.items():
    #step1: download wanted video
    orig_vid_folder=os.path.join(download_folder,'youtube_videos_original',name)
    utils.download_youtube_video(details['url'], resolution='360p', start=details['start'], num_seconds=details['len'], video_folder=orig_vid_folder)
    last_frame = [file_name for file_name in sorted(os.listdir(orig_vid_folder)) if file_name.endswith(".png")][-1]
    os.remove(os.path.join(orig_vid_folder, last_frame))

    #step2: crop if needed (removes BG in cheetah and results in more RS frames)
    if details['crop'] is not None:
        utils.crop_all_frames_in_folder(orig_vid_folder,
                                        cropped_folder=None,  #replaces in place
                                        crop_top=details['crop'][0],
                                        crop_bottom=details['crop'][1],
                                        )

    #step3: upsample by wanted factors
    for DAIN_rate in details['DAINs']:
        DAINed_folder=os.path.join(download_folder,'upsampled',f'{name}_x{DAIN_rate}')
        utils.send_DAIN_command(env_py_path=DAIN_env_python_path,
                                input_dir=orig_vid_folder,
                                output_dir=DAINed_folder,
                                num_interpolated_frames=DAIN_rate,
                                save_mp4=False,
                                save_frames=True)
        # DAIN_wrapper_for_VRS(input_dir=orig_vid_folder,
        #                      output_dir=DAINed_folder,
        #                      num_interpolated_frames=DAIN_rate,
        #                      save_mp4=False,
        #                      save_frames=True
        #                      )

        #step4: "sample" RS/GS videos. Run for GS with both center and left alignments.
        RS_folder=os.path.join(download_folder,'RS',f'{name}_x{DAIN_rate}')
        utils.synthesize_rolling_shutter(video_path=DAINed_folder,
                                         target_folder=RS_folder
                                         )
        GS_folder=os.path.join(download_folder,'GS',f'{name}_x{DAIN_rate}')
        num_rows=utils.load_image_to_torch(os.path.join(DAINed_folder,f'00000_00000.png')).shape[2]
        num_RS_frames = len([frame for frame in os.listdir(RS_folder) if frame.endswith('.png')])  #needed to not take an extra GS frame
        alignments=[{'alignment': '_left', 'start': 0, 'every':num_rows, 'num': num_RS_frames},
                    {'alignment': '_center', 'start': num_rows//2, 'every':num_rows, 'num': num_RS_frames}]
        for alignment in alignments:
            utils.sample_frames_in_folder(frames_folder=DAINed_folder,
                                          target_folder=GS_folder,
                                          sample_every=alignment['every'],
                                          start_frame=alignment['start'],
                                          num_frames=alignment['num'],
                                          rename=True,
                                          rename_suffix=alignment['alignment'],
                                          rename_restart_count=True)
        if cleanup_after:
            rmtree(DAINed_folder)
    if cleanup_after:
        rmtree(orig_vid_folder)
if cleanup_after:
    rmtree(os.path.join(download_folder,'youtube_videos_original'))

print('Done')
