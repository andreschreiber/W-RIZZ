import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import argparse
from pathlib import Path
from experiment import Experiment
from dataset import BasicDataset
from metrics import SparseDisagreementScore
from trainers import BasicTrainer, MeanTeacherTrainer
from losses import get_loss, MeanTeacherLoss
from augmentations import SparseHorizontalFlip, SparseRandomCrop, ImageOnlyAugment
from torch.utils.data import DataLoader
from torchvision import transforms


def main(settings):
    """ Runs the train code
    
    :param args_dict: command line arguments in dictionary format
    """
    
    # Create experiment
    experiment_path = Path(settings['experiment'])
    experiment = Experiment(experiment_path, should_exist=False)
    experiment.write_settings(settings)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(experiment.train_log_file), logging.StreamHandler()]
    )
    
    # Log the settings
    for key in settings:
        logging.info("Argument '{}': {}".format(key, settings[key]))
    
    # Set seed
    seed = settings['seed']
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
    rebalance_train = 'default' if settings['rebalance'] else 'none'
    save_best_model = False if settings['save_each_epoch'] else True
    use_ramping = False if settings['disable_ramping'] else True
    network_config = {'name': settings['model'], 'output_size': resolution}
    loss_config = {'name': settings['loss']}
    
    # Create evaluation metrics
    eval_metrics = {
        'disagreement(t=0.01)': SparseDisagreementScore(threshold=0.01),
        'disagreement(t=0.05)': SparseDisagreementScore(threshold=0.05),
        'disagreement(t=0.10)': SparseDisagreementScore(threshold=0.10),
        'disagreement(t=0.25)': SparseDisagreementScore(threshold=0.25),
        'disagreement(t=0.50)': SparseDisagreementScore(threshold=0.50),
    }
    
    if settings['use_mean_teacher']:
        # Mean Teacher setup
        logging.info("Preparing datasets using mean teacher approach")
        
        # Setup training data
        train_dataset = BasicDataset(
            folder_path=settings['train_folder_path'],
            csv_path=settings['train_csv_path'],
            resolution=resolution,
            augmentation=transforms.Compose([SparseHorizontalFlip(p=0.5), SparseRandomCrop(p=0.5)]),
            consistency_augmentation=transforms.RandomApply([transforms.ColorJitter((0.75, 1.75), 0.5, 0.5, 0.1)], p=0.5),
            in_memory=settings['in_memory'],
            normalize=True,
            rebalancing=rebalance_train
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=settings['batch_size'],
            num_workers=settings['num_workers'],
            shuffle=True
        )
        
        # Setup validation data (if relevant)
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
                batch_size=settings['batch_size'],
                num_workers=settings['num_workers'],
                shuffle=False
            )
        else:
            valid_dataset = None
            valid_dataloader = None
           
        # Create loss function
        loss_function = MeanTeacherLoss(
            accuracy_loss=get_loss(**loss_config),
            consistency_loss=nn.MSELoss(),
            alpha=settings['alpha'],
            beta=settings['beta']
        )
           
        # Create trainer
        trainer = MeanTeacherTrainer(
            network_config=network_config,
            loss_function=loss_function,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            eval_metrics=eval_metrics,
            pretrained_weights_file=settings['pretrained_weights'],
            save_path=experiment.model_file,
            save_best_model=save_best_model,
            lr=settings['lr'],
            lr_decay_gamma=settings['lr_decay_gamma'],
            lr_decay_step=settings['lr_decay_step'],
            weight_decay=settings['weight_decay'],
            max_alpha=settings['alpha'],
            max_beta=settings['beta'],
            use_ramping=use_ramping
        )
        # Train
        logging.info("Created network: '{}'".format(trainer.get_network().get_name()))
        trainer.train(settings['epochs'], device=device, ramp_epochs=settings['ramp_epochs'])
        logging.info("Training has concluded")
        evaluator = trainer.get_evaluator()
        
        # Evaluate on validation data
        if valid_dataloader is not None and evaluator is not None:
            # Reload checkpoint
            network = trainer.get_network()
            network.load_state_dict(torch.load(experiment.model_file))
            # Validate
            validation_results = trainer.get_evaluator().validate(network, device)
            logging.info("Saving results to {}".format(experiment.results_file))
            logging.info(validation_results)
            with open(experiment.results_file, 'w') as f:
                json.dump(validation_results, f, indent=4)
    else:
        # Default setup
        logging.info("Preparing datasets using default approach")
        train_dataset = BasicDataset(
            folder_path=settings['train_folder_path'],
            csv_path=settings['train_csv_path'],
            resolution=resolution,
            augmentation=transforms.Compose([
                SparseHorizontalFlip(p=0.5),
                SparseRandomCrop(p=0.5),
                ImageOnlyAugment(transforms.ColorJitter((0.75, 1.75), 0.5, 0.5, 0.1), p=0.5)
            ]),
            consistency_augmentation=None,
            in_memory=settings['in_memory'],
            normalize=True,
            rebalancing=rebalance_train
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=settings['batch_size'],
            num_workers=settings['num_workers'],
            shuffle=True
        )
        
        # Setup validation data if provided
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
                batch_size=settings['batch_size'],
                num_workers=settings['num_workers'],
                shuffle=False
            )
        else:
            valid_dataset = None
            valid_dataloader = None
            
        # Configure loss and trainer
        loss_function = get_loss(**loss_config)
        trainer = BasicTrainer(
            network_config=network_config,
            loss_function=loss_function,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            eval_metrics=eval_metrics,
            pretrained_weights_file=settings['pretrained_weights'],
            save_path=experiment.model_file,
            save_best_model=save_best_model,
            lr=settings['lr'],
            lr_decay_gamma=settings['lr_decay_gamma'],
            lr_decay_step=settings['lr_decay_step'],
            weight_decay=settings['weight_decay'],
        )
        # Train
        logging.info("Created network: '{}'".format(trainer.get_network().get_name()))
        trainer.train(settings['epochs'], device=device)
        logging.info("Training has concluded")
        evaluator = trainer.get_evaluator()
        
        # Evaluate on validation data
        if valid_dataloader is not None and evaluator is not None:
            # Reload checkpoint
            network = trainer.get_network()
            network.load_state_dict(torch.load(experiment.model_file))
            # Validate
            validation_results = trainer.get_evaluator().validate(network, device)
            logging.info("Saving results to {}".format(experiment.results_file))
            logging.info(validation_results)
            with open(experiment.results_file, 'w') as f:
                json.dump(validation_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed to use
    parser.add_argument("--seed", type=int, default=123)
    
    # Data configurations
    parser.add_argument("--train_folder_path", type=str, required=True)
    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--valid_folder_path", type=str, required=False)
    parser.add_argument("--valid_csv_path", type=str, required=False)
    
    # Experiment path to save to
    parser.add_argument("--experiment", type=str)
    
    # Resolution
    parser.add_argument("--res_x", type=int, default=424)
    parser.add_argument("--res_y", type=int, default=240)
    
    # Data loading
    parser.add_argument("--batch_size", type=int, default=8) # each batch actually has 2 images (so batch_size=8 => 16 images in batch)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--in_memory", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rebalance", default=False, action=argparse.BooleanOptionalAction)
    
    # Model
    parser.add_argument("--model", type=str, default="travnetup3nnrgb")
    parser.add_argument("--pretrained_weights", type=str)
    
    # Optimizer settings
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--lr_decay_step", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0)
    
    # Mean teacher settings
    parser.add_argument("--use_mean_teacher", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable_ramping", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    
    # Training
    parser.add_argument("--save_each_epoch", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ramp_epochs", type=int, default=10) # irrelevant if mean teacher not enabled
    parser.add_argument("--loss", type=str, default="lrizz")
    
    args = parser.parse_args()
    main(vars(args)) # pass args as dictionary
