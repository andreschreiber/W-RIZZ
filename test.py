import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import argparse
from pathlib import Path
from networks import get_network
from experiment import Experiment
from dataset import BasicDataset
from metrics import SparseDisagreementScore
from evaluators import BasicEvaluator
from torch.utils.data import DataLoader
from torchvision import transforms


def main(args_dict):
    """ Runs the test code
    
    :param args_dict: command line arguments in dictionary format
    """
    
    # Get the experiment
    experiment_path = Path(args_dict['experiment'])
    experiment = Experiment(experiment_path, should_exist=True)
    settings = experiment.read_settings()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(experiment.test_log_file), logging.StreamHandler()]
    )
    
    # Log the settings
    for key in settings:
        logging.info("Setting '{}': {}".format(key, settings[key]))
    
    # Log the arguments
    for key in args_dict:
        logging.info("Cmd Arg '{}': {}".format(key, args_dict[key]))
    
    # Set seed
    seed = args_dict['seed']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse relevant arguments
    resolution = (settings['res_y'], settings['res_x'])
    network_config = {'name': settings['model'], 'output_size': resolution}
    
    # Create evaluation metrics
    eval_metrics = {
        'disagreement(t=0.10)': SparseDisagreementScore(threshold=0.10),
        'disagreement(t=0.25)': SparseDisagreementScore(threshold=0.25),
        'disagreement(t=0.50)': SparseDisagreementScore(threshold=0.50),
        
        'disagreement,eq(t=0.10)': SparseDisagreementScore(threshold=0.10, mode='eq'),
        'disagreement,eq(t=0.25)': SparseDisagreementScore(threshold=0.25, mode='eq'),
        'disagreement,eq(t=0.50)': SparseDisagreementScore(threshold=0.50, mode='eq'),
        
        'disagreement,neq(t=0.10)': SparseDisagreementScore(threshold=0.10, mode='neq'),
        'disagreement,neq(t=0.25)': SparseDisagreementScore(threshold=0.25, mode='neq'),
        'disagreement,neq(t=0.50)': SparseDisagreementScore(threshold=0.50, mode='neq'),
    }
    
    # Create network
    network = get_network(**network_config)
    network.load_state_dict(torch.load(experiment.model_file))
    
    # Results
    test_results = {}
    
    # Compute results on train dataset
    logging.info("Preparing train dataset")
    train_dataset = BasicDataset(
        folder_path=settings['train_folder_path'],
        csv_path=settings['train_csv_path'],
        resolution=resolution,
        augmentation=None,
        consistency_augmentation=None,
        in_memory=settings['in_memory'],
        normalize=True,
        rebalancing='none'
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=settings['num_workers'],
        shuffle=False
    )
    logging.info("Running evaluation on train dataset")
    train_evaluator = BasicEvaluator(train_dataloader, eval_metrics)
    test_results['train'] = train_evaluator.validate(network, device)

    # Compute results on validation set (if provided)
    if settings['valid_folder_path'] and settings['valid_csv_path']:
        valid_dataset = BasicDataset(
            folder_path=settings['valid_folder_path'],
            csv_path=settings['valid_csv_path'],
            resolution=resolution,
            augmentation=None,
            consistency_augmentation=None,
            in_memory=settings['in_memory'],
            normalize=True,
            rebalancing='none'
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=settings['num_workers'],
            shuffle=False
        )
        
        logging.info("Running evaluation on validation dataset")
        validation_evaluator = BasicEvaluator(valid_dataloader, eval_metrics)
        test_results['validation'] = validation_evaluator.validate(network, device)
    
    # Print and save results
    logging.info(test_results)
    if args_dict['output']:
        logging.info("Saving results to {}".format(args_dict['output']))
        with open(args_dict['output'], 'w') as f:
            json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed to use
    parser.add_argument("--seed", type=int, default=123)
    # Experiment path
    parser.add_argument("--experiment", type=str)
    # Where to save output
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    main(vars(args)) # pass args as dictionary
