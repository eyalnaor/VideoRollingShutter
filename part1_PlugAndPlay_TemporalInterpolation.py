"""
Part 1 for running the 2022 ECCV paper:
"Combining Internal and External Constraints for Unrolling Shutter in Videos" [1].
This part is the plug-and-play external temporal interpolation (TI) algorithm, detailed in section 4.1 in the paper.
Here we use (as in the paper) - DAIN [2].
This part is separate in order to both emphasize the plug-and-play nature of the temporal interpolation module in our
pipeline, and to allow for different python environments for the external TI code and our code.
This script requires DAIN's environment as detailed in their git page: https://github.com/baowenbo/DAIN,
and of course running part2_MergeNet_and_xtOptimization.py after this step.

In this step:
0. User provides RS video and wanted (trained) MergeNet.
1. Using an external out-of-the-box frame-interpolation method DAIN [1], we interpolate between the RS frames by the
   factor of the number of rows. This is seen as the "continuous" space-time volume.
   We do so with 16 augmentations of flips and rotations.
   We note that this step is plug-and-play, and can be replaces with other frame-interpolation methods, thus
   gaining performance boosts for the RS problem when advances in frame-interpolation are made.
   Note that this step can be run on parallel machines for x16 speedup, depending on your setup

[1] @inproceedings{naor2022videors,
  title={Combining Internal and External Constraints for Unrolling Shutter in Videos},
  author={Naor, Eyal and Antebi, Itai and Bagon, Shai and Irani, Michal},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

[2] @inproceedings{bao2019dain,
  title={Depth-aware video frame interpolation},
  author={Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

      If you plan on quoting this paper, please quote [2] as well.
"""
import os
import utils
import config_utils
from DAIN.DAIN_for_VideoRollingShutter import DAIN_wrapper_for_VRS

parser = config_utils.create_parser()
args = parser.parse_args()
config = config_utils.startup(json_path=args.config, args=args)

input_RS_father_folder=config['input_RS']['input_RS_folder']
vid_folders = sorted([dir for dir in os.listdir(input_RS_father_folder) if
                      os.path.isdir(os.path.join(input_RS_father_folder, dir))])  # internal dirs
for vid_name in vid_folders:
    # step1: external temporal-interpolation method (DAIN) on input video
    RS_video_folder = os.path.join(input_RS_father_folder, vid_name)
    RS_in_name = config['input_RS']['input_in_name']
    num_rows = utils.get_num_rows(RS_video_folder, RS_in_name)
    interpolated_folder_main = os.path.join(config['save_dir'], vid_name, 'after_DAIN')
    MergeNet_folder = os.path.join(config['save_dir'], vid_name, 'after_MergeNet')

    # Run DAIN on all augmentations. Note that this step can be run on parallel machines for x16 speedup, depending on your setup
    for reverse_ in [False, True]:
        for rot_ in [0, 90, 180, 270]:
            for flip_ in ['none', 'hor']:
                if config["verbose"]:
                    print(
                        f'****************entered aug: reverse_: {reverse_}, rot_: {rot_}, flip_: {flip_}****************')
                interpolated_folder_aug = os.path.join(interpolated_folder_main,
                                                       f'reverse{reverse_}_rot{rot_}_flip{flip_}')

                DAIN_wrapper_for_VRS(input_dir=RS_video_folder,
                                     output_dir=interpolated_folder_aug,
                                     input_in_name=RS_in_name,
                                     num_interpolated_frames=num_rows,
                                     reverse_order=reverse_,
                                     rotation=rot_,
                                     flip=flip_,
                                     save_mp4=True,  # faster save and load for mp4 than frames
                                     save_frames=False,
                                     verbose=config["verbose"]
                                     )
    if config["verbose"]:
        print(f'***** finished temporal-interpolation on 16 augmentations for video: {vid_name} *****')

if config["verbose"]:
    print(f'Finished part 1 (plug-and-play temporal interpolation). Now run part2_MergeNet_and_xtOptimization.py')