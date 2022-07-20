"""
Part 2 for running the 2022 ECCV paper:
"Combining Internal and External Constraints for Unrolling Shutter in Videos" [1].
This part is our written algorithm, detailed in section 4 in the paper.
Part 1 is separate in order to both emphasize the plug-and-play nature of the temporal interpolation module in our
pipeline [2], and to allow for different python environments for the external TI code and our code.
This script requires our environment detailed in the Readme, and of course pre-running part1 with the same config file.
In this step:
2. We "sample" the "continuous" space-time volume in the appropriate locations to generate the 16 suggestions for
   GS solutions.
3. Using a trained MergeNet (specified by user), we merge these 16 suggestions to a single one.
   We note that training MergeNet is also made possible, see relevant folder.
4. We apply an additional zero-shot optimization which imposes the similarity of xt-patches between the GS output video
   and the RS input video.
   The parameters in this step are also configurable, and may be optimized for different datasets. See relevant folder.


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
from shutil import rmtree
import utils
import config_utils
import MergeNet.MergeNet_main as MergeNet
import xt_optimization.xt_optimization_main as xt_optimization

parser = config_utils.create_parser()
args = parser.parse_args()
config = config_utils.startup(json_path=args.config, args=args)

input_RS_father_folder=config['input_RS']['input_RS_folder']
vid_folders = sorted([dir for dir in os.listdir(input_RS_father_folder) if
                      os.path.isdir(os.path.join(input_RS_father_folder, dir))])  # internal dirs
for vid_name in vid_folders:
    # step1: external temporal-interpolation method (DAIN) on input video
    RS_video_folder = os.path.join(input_RS_father_folder, vid_name)

    interpolated_folder_main = os.path.join(config['save_dir'], vid_name, 'after_DAIN')
    GS_suggestion_main = os.path.join(config['save_dir'], vid_name, 'GS_suggestions')
    MergeNet_folder = os.path.join(config['save_dir'], vid_name, 'after_MergeNet')

    for reverse_ in [False, True]:
        for rot_ in [0, 90, 180, 270]:
            for flip_ in ['none', 'hor']:
                interpolated_folder_aug = os.path.join(interpolated_folder_main,
                                                       f'reverse{reverse_}_rot{rot_}_flip{flip_}')
                assert os.path.isdir(interpolated_folder_aug),\
                    f'Temporally interpolated folder {interpolated_folder_aug} does not exist. Run part1_PlugAndPlay_TemporalInterpolation.py'

                GS_suggestion_output_folder_aug = os.path.join(GS_suggestion_main,
                                                    f'reverse{reverse_}_rot{rot_}_flip{flip_}_GS_suggestion')
                utils.GS_from_dense_RS_folder_vids(dense_RS_folder=interpolated_folder_aug,
                                                   GS_output_folder=GS_suggestion_output_folder_aug,
                                                   RS_direction='ver',
                                                   GS_frames_to_make=None,  # does all frames possible
                                                   GS_centered_around='left'
                                                   )
                # cleanup if needed - delete the dense space-time volume
                if config['cleanup_during_run']:
                    rmtree(interpolated_folder_aug)

    #step3: run MergeNet to merge the 16 suggestions
    # Also "crop" the config section for MergeNet. This is since MergeNet can also be run separately.
    # Need to also add some paths.
    MergeNet_config = config['MergeNet_config']
    MergeNet_config['save_dir'] = os.path.join(config['save_dir'], 'MergeNet')
    MergeNet_config['debug'] = config['debug']
    MergeNet.main(MergeNet_config, eval_single_folder=GS_suggestion_main)

    # #step4: run the test-time video-specific optimization on xt patches
    # Also "crop" the config section for xt_optimization. This is since this step can also be run separately.
    xt_optimization_config = config['xt_optimization_config']
    xt_optimization_config['save_dir'] = os.path.join(config['save_dir'], 'xt_optimization')
    xt_optimization_config['debug'] = config['debug']
    xt_optimization.xt_optimization_single(xt_optimization_config, MergeNet_folder, RS_video_folder,
                                           save_folder=None, GT_path=None, gt_in_name=None, gt_direction=config['GS_centered_around'])

print('done.')