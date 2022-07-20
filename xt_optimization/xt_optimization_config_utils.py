import json
import os
import shutil
from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to json config file', default='config.json')
    parser.add_argument('-de', '--debug', type=eval, help='debug - will not save results', default=None)

    return parser

def read_json_with_line_comments(cjson_path):
    with open(cjson_path, 'r') as R:
        valid = []
        for line in R.readlines():
            if line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            valid.append(line)
    return json.loads(' '.join(valid))


def startup(json_path, args=None, copy_files=True):
    print('-startup- reading config json {}'.format(json_path))
    config = read_json_with_line_comments(json_path)
    if args is not None and hasattr(args, 'debug') and args.debug is not None:
        config['debug'] = args.debug

    if config['debug']:
        working_dir = None
        print(f'*******DEBUG MODE - no working dir*******\n'*5)
    else:
        working_dir = config["working_dir"]

    #overrun input folders if instructed. Enables switching data with single entry, but still a little flexible for other datasets
    if config['overrun_dataset_alignment'] is not None:
        if 'fastec_center' in config['overrun_dataset_alignment']:
            config['dataset'] = 'fastec'
            config['alignment'] = 'center'
            config['video_name'] = None if 'fastec_center' == config['overrun_dataset_alignment'] else config['overrun_dataset_alignment'].split('^')[-1]
        elif 'fastec_left' in config['overrun_dataset_alignment']:
            config['dataset'] = 'fastec'
            config['alignment'] = 'left'
            config['video_name'] = None if 'fastec_left' == config['overrun_dataset_alignment'] else config['overrun_dataset_alignment'].split('^')[-1]
        elif 'carla_center' in config['overrun_dataset_alignment']:
            config['dataset'] = 'carla'
            config['alignment'] = 'center'
            config['video_name'] = None if 'carla_center' == config['overrun_dataset_alignment'] else config['overrun_dataset_alignment'].split('^')[-1]
        elif 'carla_left' in config['overrun_dataset_alignment']:
            config['dataset'] = 'carla'
            config['alignment'] = 'left'
            config['video_name'] = None if 'carla_left' == config['overrun_dataset_alignment'] else config['overrun_dataset_alignment'].split('^')[-1]
        elif 'spinners_left' in config['overrun_dataset_alignment']:
            config['dataset'] = 'spinners'
            config['alignment'] = 'left'
            config['video_name'] = None
        elif 'spinners_center' in config['overrun_dataset_alignment']:
            config['dataset'] = 'spinners'
            config['alignment'] = 'center'
            config['video_name'] = None
        elif 'youtube_left' in config['overrun_dataset_alignment']:
            config['dataset'] = 'youtube'
            config['alignment'] = 'left'
            config['video_name'] = None
        elif 'youtube_center' in config['overrun_dataset_alignment']:
            config['dataset'] = 'youtube'
            config['alignment'] = 'center'
            config['video_name'] = None
        elif 'youtube2_left' in config['overrun_dataset_alignment']:
            config['dataset'] = 'youtube2'
            config['alignment'] = 'left'
            config['video_name'] = None
        elif 'youtube2_center' in config['overrun_dataset_alignment']:
            config['dataset'] = 'youtube2'
            config['alignment'] = 'center'
            config['video_name'] = None
        else: assert False, f'unknown config["overrun_dataset_alignment"]: {config["overrun_dataset_alignment"]}'
        print(f'***** Overrunning config dataset/alignment with config["overrun_dataset_alignment"]: {config["overrun_dataset_alignment"]}')


    # we want to save and save folder already exists, find non-existing folder
    if working_dir is not None and os.path.isdir(working_dir):
        v = 0
        while True:
            working_dir_v = working_dir + '{}-v{}'.format(config['tag'], v)
            if not os.path.isdir(working_dir_v):
                break
            v += 1
        config['save_dir'] = working_dir_v
    else:
        config['save_dir'] = working_dir
    if config['save_dir'] is not None:
        os.makedirs(config['save_dir'], exist_ok=True)
    print(f'*** save_dir: {config["save_dir"]} ***')

    # copy files?
    if copy_files and working_dir is not None:
        code_folder = os.path.join(config['save_dir'], 'code')
        os.makedirs(code_folder, exist_ok=True)
        for filename in os.listdir('.'):
            if filename.endswith('.py'):
                shutil.copy(filename, code_folder)
            shutil.copy(json_path, code_folder)
        with open(os.path.join(code_folder, '_processed_config.json'), 'w') as W:
            W.write(json.dumps(config, indent=2))

    # assertions and additions
    # if 'carla' in os.path.split(config['training_input_father_dir'])[-1]:
    #     dataset='carla'
    # elif 'fastec' in os.path.split(config['training_input_father_dir'])[-1]:
    #     dataset='fastec'
    # else: assert False, f'unknown dataset'
    # if config['training_input_in_name'] == 'left':
    #     if dataset=='fastec': config['training_gt_in_name'] = 'first.png'
    #     elif dataset=='carla': config['training_gt_in_name'] = 'gs_f.png'
    #     else: assert False
    # elif config['training_input_in_name'] == 'center':
    #     if dataset=='fastec': config['training_gt_in_name'] = 'middle.png'
    #     elif dataset=='carla': config['training_gt_in_name'] = 'gs_m.png'
    #     else: assert False
    # else: assert False, f"unknown training_input_in_name: {config['training_input_in_name']}"

    return config
