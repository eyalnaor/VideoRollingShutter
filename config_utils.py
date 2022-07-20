"""
separate from utils for clean order
"""

from argparse import ArgumentParser
import json
import os
import shutil

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to json config file', default='config.json')
    return parser




def read_json_with_line_comments(cjson_path):
    with open(cjson_path, 'r') as R:
        valid = []
        for line in R.readlines():
            if line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            valid.append(line)
    return json.loads(' '.join(valid))



def startup(json_path, args=None):
    """
    startups and parses the config file.
    """
    print('-startup- reading config json {}'.format(json_path))
    config = read_json_with_line_comments(json_path)

    if args is not None and hasattr(args, 'debug') and args.debug is not None:
        config['debug'] = args.debug

    if config['debug']:
        working_dir = None
        print(f'*******DEBUG MODE - no working dir*******\n'*5)
    else:
        working_dir = config["working_dir"]

    config['save_dir'] = working_dir
    if config['save_dir'] is not None:
        os.makedirs(config['save_dir'], exist_ok=True)
    print(f'*** save_dir: {config["save_dir"]} ***')

    # # copy files?
    # if copy_files and working_dir is not None:
    #     code_folder = os.path.join(config['save_dir'], 'code')
    #     os.makedirs(code_folder, exist_ok=True)
    #     for filename in os.listdir('.'):
    #         if filename.endswith('.py'):
    #             shutil.copy(filename, code_folder)
    #         shutil.copy(json_path, code_folder)
    #     with open(os.path.join(code_folder, '_processed_config.json'), 'w') as W:
    #         W.write(json.dumps(config, indent=2))

    return config