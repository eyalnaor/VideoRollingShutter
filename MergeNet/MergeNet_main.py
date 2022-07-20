import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import Network
import utils
import config_utils
import cProfile
import io
import pstats

import data_handler
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main(config, eval_single_folder=None):
    """
    eval_single_folder: enables easier use with VideoRS_Main
    """

    # train if needed (no ckpt, or want to do other than only eval/fine_tune)
    need_to_train = not config["loading_model"]["checkpoint"] or not (config["loading_model"]["only_eval"] or config["loading_model"]["only_fine_tune"])
    config['need_to_train']=need_to_train  # needed internally - whether to save tensorboard or not
    params = {'batch_size': config["batch_size"] if hasattr(config,"batch_size") else 1,
              'shuffle': True,
              'num_workers': 0,
              'worker_init_fn': worker_init_fn}

    # initialize the net
    network = Network.mergenet(config=config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # need to initialize anyway, even if loading ckpt
    if config["loading_model"]["checkpoint"] != "":  # checkpoint given, load
        network.load_model(config["loading_model"]["checkpoint"])

    if need_to_train:
        dataset = data_handler.merge_DataHandler(config=config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        data_generator = data.DataLoader(dataset, **params)
        network.train(data_generator)

    eval_TestSet = not need_to_train or config["test_set"]["eval_on_test_at_end"]  # if not trained, want to eval. Also, if want to save test results

    # run on test set
    if eval_TestSet:
        if eval_single_folder is not None:
            network.eval_on_single(eval_single_folder)
        else:  # run on the test set from the config
            save_results_testSet = config["test_set"]["save_test_images"]
            test_set_diff_loss, test_set_diff_psnr, test_set_diff_ssim = network.results_on_TestSet(config, save_results_testSet)






if __name__ == '__main__':
    # Need to generate config if running only MergeNet, else gets config externally.
    parser = config_utils.create_parser()
    args = parser.parse_args()
    config = config_utils.startup(json_path=args.config, args=args)
    main(config)
