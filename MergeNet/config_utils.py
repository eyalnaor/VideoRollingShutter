import json
import os
import shutil
import utils
import copy

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

def edit_config_for_fine_tune(config):
    "overruns config's training configuration with the finetuning configuration"
    fine_tune_config = copy.deepcopy(config)
    fine_tune_config['training_input_father_dir']=fine_tune_config['fine_tuning']['input_dir']
    fine_tune_config['training_input_in_name']=fine_tune_config['fine_tuning']['input_in_name']
    fine_tune_config['training_gt_father_dir']=fine_tune_config['fine_tuning']['gt_dir']
    fine_tune_config['training_gt_in_name']=fine_tune_config['fine_tuning']['gt_in_name']
    fine_tune_config['datasets']=fine_tune_config['fine_tuning']['fine_tune_datasets']
    fine_tune_config['preload_training_data']=False
    fine_tune_config['preload_val_data']=False
    fine_tune_config['datasets_ratios'] = [1 / len(fine_tune_config['training_input_father_dir']) for i in range(len(fine_tune_config['training_input_father_dir']))]
    return fine_tune_config

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

    if config['datasets_ratios'] is None:  # assign uniform ratio to all datasets
        config['datasets_ratios'] = [1/len(config['training_input_father_dir']) for i in range(len(config['training_input_father_dir']))]

    if config['data_augmenter']['augs_and_ensemble'] is not None:
        if config['data_augmenter']['augs_and_ensemble'] == 'only_augs':
            config['data_augmenter']['flip_hor_prob'] = 0.5
            config['data_augmenter']['flip_ver_prob'] = 0.0
            config['data_augmenter']['rot_prob'] = 0.5
            config['data_augmenter']['eval_ensemble'] = False
            config['data_augmenter']['eval_ensemble_not_mixing_xy'] = True
        elif config['data_augmenter']['augs_and_ensemble'] == 'only_eval':
            config['data_augmenter']['flip_hor_prob'] = 0.0
            config['data_augmenter']['flip_ver_prob'] = 0.0
            config['data_augmenter']['rot_prob'] = 0.0
            config['data_augmenter']['eval_ensemble'] = True
            config['data_augmenter']['eval_ensemble_not_mixing_xy'] = True
        elif config['data_augmenter']['augs_and_ensemble'] == 'augs_and_ensemble':
            config['data_augmenter']['flip_hor_prob'] = 0.5
            config['data_augmenter']['flip_ver_prob'] = 0.0
            config['data_augmenter']['rot_prob'] = 0.75
            config['data_augmenter']['eval_ensemble'] = True
            config['data_augmenter']['eval_ensemble_not_mixing_xy'] = True
        elif config['data_augmenter']['augs_and_ensemble'] == 'no_augs_no_ensemble':
            config['data_augmenter']['flip_hor_prob'] = 0.0
            config['data_augmenter']['flip_ver_prob'] = 0.0
            config['data_augmenter']['rot_prob'] = 0.0
            config['data_augmenter']['eval_ensemble'] = False
            config['data_augmenter']['eval_ensemble_not_mixing_xy'] = True

        else: assert False, f'unknown val for {config["data_augmenter"]["augs_and_ensemble"]}: {config["data_augmenter"]["augs_and_ensemble"]}'

    if config['fine_tuning']['fine_tune_datasets'] is not None:
        config['fine_tuning']['fine_tune'] = True
        config['fine_tuning']['input_dir'] = []
        config['fine_tuning']['input_in_name'] = []
        config['fine_tuning']['gt_dir'] = []
        config['fine_tuning']['gt_in_name'] = []
        config['test_set']['test_input_father_dir']=[]
        config['test_set']['test_gt_father_dir']=[]


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
    print(f'*** working_dir: {config["save_dir"]} ***')

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
    config['training_gt_in_name']=[]  # list to support multiple datasets
    for (training_input_father_dir_single_dataset, training_input_in_name_single_dataset) in zip(
            config['training_input_father_dir'], config['training_input_in_name']):

        if 'carla' in os.path.split(training_input_father_dir_single_dataset)[-1]:
            dataset='carla'
        elif 'fastec' in os.path.split(training_input_father_dir_single_dataset)[-1]:
            dataset='fastec'
        else:
            dataset = None

        _, gt_in_name = get_in_names(training_input_in_name_single_dataset, dataset)
        config['training_gt_in_name'].append(gt_in_name)
    return config

def get_in_names(input_in_name, dataset):
    if input_in_name=='center' and dataset=='fastec':
        return 'center', 'middle.png'
    elif input_in_name=='center' and dataset=='carla':
        return 'center', 'gs_m.png'
    elif input_in_name=='left' and dataset=='fastec':
        return 'left', 'first.png'
    elif input_in_name=='left' and dataset=='carla':
        return 'left', 'gs_f.png'
    else: return input_in_name, input_in_name
