import tqdm
import time
import math
import torch
import logging
import networks
import numpy as np
from evaluators import BasicEvaluator
from torch_ema import ExponentialMovingAverage


class Trainer:
    """ Base class for trainers """
    def __init__(self,
                 network_config,
                 loss_function,
                 train_dataloader,
                 valid_dataloader,
                 eval_metrics=None,
                 pretrained_weights_file=None,
                 save_path=None,
                 save_best_model=False,
                 lr=0.001,
                 lr_decay_gamma=0.1,
                 lr_decay_step=10,
                 weight_decay=0):
        """ Creates trainer
        
        :param network_config: configuration for neural network (dictionary)
        :param loss_function: loss function to use
        :param train_dataloader: train dataloader
        :param valid_dataloader: validation dataloader
        :param eval_metrics: evaluation metrics to use
        :param pretrained_weights_file: file for pretrained weights (can be None)
        :param save_path: path to save model to
        :param save_best_model: if True, the best model is saved (rather than saving each epoch)
        :param lr: learning rate
        :param lr_decay_gamma: lr decay gamma for learning rate decay
        :param lr_decay_step: number of epochs between learning rate decreases
        :param weight_decay: weight decay factor
        """
        
        self._network_config = network_config
        self._loss_function = loss_function
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._eval_metrics = eval_metrics
        self._pretrained_weights_file = pretrained_weights_file
        self._save_path = save_path
        self._save_best_model = save_best_model
        self._lr = lr
        self._lr_decay_gamma = lr_decay_gamma
        self._lr_decay_step = lr_decay_step
        self._weight_decay = weight_decay
        
        if self._save_best_model and self._save_path is None:
            raise ValueError("save_best_model=True but save_path=None")
        
        if self._save_best_model and self._eval_metrics is None:
            raise ValueError("save_best_model=True but eval_metrics=None")

    def train(self, epochs):
        raise NotImplementedError
        
    def get_network(self):
        raise NotImplementedError
        
    def get_evaluator(self):
        return self._evaluator

#
# Trainer for Basic Setup
#

class BasicTrainer(Trainer):
    """ Basic (non-mean teacher) trainer """
    def __init__(self,
                 network_config,
                 loss_function,
                 train_dataloader,
                 valid_dataloader,
                 eval_metrics=None,
                 pretrained_weights_file=None,
                 save_path=None,
                 save_best_model=False,
                 lr=0.001,
                 lr_decay_gamma=0.1,
                 lr_decay_step=10,
                 weight_decay=0):
        """ Creates basic trainer
        
        :param network_config: configuration for neural network (dictionary)
        :param loss_function: loss function to use
        :param train_dataloader: train dataloader
        :param valid_dataloader: validation dataloader
        :param eval_metrics: evaluation metrics to use
        :param pretrained_weights_file: file for pretrained weights (can be None)
        :param save_path: path to save model to
        :param save_best_model: if True, the best model is saved (rather than saving each epoch)
        :param lr: learning rate
        :param lr_decay_gamma: lr decay gamma for learning rate decay
        :param lr_decay_step: number of epochs between learning rate decreases
        :param weight_decay: weight decay factor
        """
        
        super().__init__(
            network_config,
            loss_function,
            train_dataloader,
            valid_dataloader,
            eval_metrics,
            pretrained_weights_file,
            save_path,
            save_best_model,
            lr,
            lr_decay_gamma,
            lr_decay_step,
            weight_decay
        )
        
        # Create network
        self._network = networks.get_network(**network_config)
        # Load pretrained weights (e.g., from RUGD segmentation training) if specified
        if self._pretrained_weights_file is not None:
            logging.info("Loading weights from {}".format(self._pretrained_weights_file))
            self._network.load_pretrained(torch.load(self._pretrained_weights_file))
        
        # Create optimizer and scheduler
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=self._lr_decay_step, gamma=self._lr_decay_gamma)
        if valid_dataloader is not None and self._eval_metrics is not None:
            self._evaluator = BasicEvaluator(valid_dataloader, self._eval_metrics)
        else:
            self._evaluator = None

    def get_network(self):
        """ Returns the network used by this trainer """
        return self._network
    
    def train(self, epochs, device):
        """ Train model 
        
        :param epochs: number of epochs to train for
        :param device: device to use
        """
        
        best_metric_epoch = 0
        min_metric = float('inf')
        
        # Train for the desired number of epochs
        for epoch in range(1, epochs+1):
            elapsed = time.time()
            mean_loss = self.train_epoch(device)
            if self._evaluator is None:
                validation_results = None
            else:
                validation_results = self._evaluator.validate(self._network, device) # validate if needed
            elapsed = time.time() - elapsed
            
            logging.info("Finished epoch {} in {:.2f} sec with loss = {:.4f}".format(
                epoch, elapsed, mean_loss
            ))
            
            if validation_results is not None:
                # Print epoch validation results
                validation_results_str = ""
                for k in validation_results.keys():
                    validation_results_str += "{}={:.4f}, ".format(k, validation_results[k])
                validation_results_str = validation_results_str[:-2]
                logging.info("Validation results: {}".format(validation_results_str))
                
                # Save/print if new best
                metrics_list = [(validation_results[k], k) for k in validation_results.keys()]
                step_min_metric = metrics_list[int(np.argmin([v[0] for v in metrics_list]))]
                if step_min_metric[0] < min_metric:
                    logging.info("Best metric found: {}={:.4f}".format(step_min_metric[1], step_min_metric[0]))
                    best_metric_epoch = epoch
                    min_metric = step_min_metric[0]
                    if self._save_best_model:
                        logging.info("Saving model")
                        torch.save(self._network.state_dict(), self._save_path)
            
            if not self._save_best_model and self._save_path is not None:
                logging.info("Saving model")
                torch.save(self._network.state_dict(), self._save_path)
                
            self._scheduler.step()
        
        logging.info("Training concluded (best metric={:.4f} at epoch={})".format(min_metric, best_metric_epoch))
    
    def train_epoch(self, device):
        """ Train for a single epoch
        
        :param device: device to use
        :returns: mean loss for the epoch
        """
        self._network.to(device)
        self._network.train()
        losses = []
        for item in tqdm.tqdm(self._train_dataloader):
            B = item[0].shape[0]
            imageA = item[0].to(device)
            imageB = item[1].to(device)
            label = item[-1].to(device)
            self._optimizer.zero_grad()
            predictions = self._network(torch.stack([imageA, imageB], dim=1).flatten(0,1))
            predictions = {k: predictions[k].unflatten(0, (B,2)) for k in predictions.keys()}
            step_loss = self._loss_function(predictions=predictions, targets=label)
            step_loss.backward()
            self._optimizer.step()
            losses.append(step_loss.detach().cpu().item())
        return np.array(losses).mean()

#
# Trainer for Mean Teacher Setup
#

class MeanTeacherTrainer(Trainer):
    """ Mean teacher trainer """
    def __init__(self,
                 network_config,
                 loss_function,
                 train_dataloader,
                 valid_dataloader,
                 eval_metrics=None,
                 pretrained_weights_file=None,
                 save_path=None,
                 save_best_model=False,
                 lr=0.001,
                 lr_decay_gamma=0.1,
                 lr_decay_step=10,
                 weight_decay=0,
                 max_alpha=1.0,
                 max_beta=10.0,
                 use_ramping=True):
        """ Creates mean teacher trainer
        
        :param network_config: configuration for neural network (dictionary)
        :param loss_function: loss function to use
        :param train_dataloader: train dataloader
        :param valid_dataloader: validation dataloader
        :param eval_metrics: evaluation metrics to use
        :param pretrained_weights_file: file for pretrained weights (can be None)
        :param save_path: path to save model to
        :param save_best_model: if True, the best model is saved (rather than saving each epoch)
        :param lr: learning rate
        :param lr_decay_gamma: lr decay gamma for learning rate decay
        :param lr_decay_step: number of epochs between learning rate decreases
        :param weight_decay: weight decay factor
        :param max_alpha: maximum alpha for mean teacher loss
        :param max_beta: maximum beta for mean teacher loss
        :param use_ramping: whether to use a mean teacher ramping/warmup
        """
        
        super().__init__(
            network_config,
            loss_function,
            train_dataloader,
            valid_dataloader,
            eval_metrics,
            pretrained_weights_file,
            save_path,
            save_best_model,
            lr,
            lr_decay_gamma,
            lr_decay_step,
            weight_decay
        )
        
        self._max_alpha = max_alpha
        self._max_beta = max_beta
        self._use_ramping = use_ramping
        
        # Create student
        self._student_network = networks.get_network(**network_config)
        if self._pretrained_weights_file is not None:
            # load weights for student if needed
            logging.info("Loading weights from {}".format(self._pretrained_weights_file))
            self._student_network.load_pretrained(torch.load(self._pretrained_weights_file))
        # Create teacher
        self._teacher_network = networks.freeze_weights(networks.get_network(**network_config))
        self._teacher_network.load_state_dict(self._student_network.state_dict())
        self._ema = ExponentialMovingAverage(self._teacher_network.parameters(), decay=0.99, use_num_updates=True)
        
        # Create optimizer and scheduler
        self._optimizer = torch.optim.Adam(self._student_network.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=self._lr_decay_step, gamma=self._lr_decay_gamma)
        if valid_dataloader is not None and self._eval_metrics is not None:
            self._evaluator = BasicEvaluator(valid_dataloader, self._eval_metrics)
        else:
            self._evaluator = None

    def get_network(self):
        """ Return the teacher network """
        return self._teacher_network
    
    def train(self, epochs, device, ramp_epochs=None):
        """ Train using mean teacher style setup
        
        :param epochs: number of epochs to train for
        :param device: device to use
        :param ramp_epochs: number of ramping epochs (or None if use default)
        """
        
        if ramp_epochs is None: # default ramping is epochs/4
            ramp_epochs = epochs // 4
        logging.info("Using {} ramping epochs".format(ramp_epochs))
        
        best_metric_epoch = 0
        min_metric = float('inf')
        
        # Train for desired number of epochs
        for epoch in range(1, epochs+1):
            if self._use_ramping:
                # Apply ramping function
                self._loss_function.alpha = self._max_alpha
                if epoch <= ramp_epochs:
                    self._loss_function.beta = self._max_beta * math.exp(-5*(1 - epoch / ramp_epochs)**2)
                    self._ema.decay = 0.99
                else:
                    self._loss_function.beta = self._max_beta
                    self._ema.decay = 0.999
            else:
                # Use default non-ramping settings
                self._loss_function.alpha = self._max_alpha
                self._loss_function.beta = self._max_beta
                self._ema.decay = 0.99
            
            # Train
            elapsed = time.time()
            mean_loss = self.train_epoch(device)
            if self._evaluator is None:
                validation_results = None
            else:
                validation_results = self._evaluator.validate(self._teacher_network, device)
            elapsed = time.time() - elapsed
            
            logging.info("Finished epoch {} in {:.2f} seconds with loss = {:.4f}".format(
                epoch, elapsed, mean_loss
            ))
            
            if validation_results is not None:
                # Print validation results
                validation_results_str = ""
                for k in validation_results.keys():
                    validation_results_str += "{}={:.4f}, ".format(k, validation_results[k])
                validation_results_str = validation_results_str[:-2]
                logging.info("Validation results: {}".format(validation_results_str))
                
                # Save/print if new best
                metrics_list = [(validation_results[k], k) for k in validation_results.keys()]
                step_min_metric = metrics_list[int(np.argmin([v[0] for v in metrics_list]))]
                if step_min_metric[0] < min_metric:
                    logging.info("Best metric found {}={:.4f}".format(step_min_metric[1], step_min_metric[0]))
                    best_metric_epoch = epoch
                    min_metric = step_min_metric[0]
                    if self._save_best_model:
                        logging.info("Saving model")
                        torch.save(self._teacher_network.state_dict(), self._save_path)
            
            if not self._save_best_model and self._save_path is not None:
                logging.info("Saving model")
                torch.save(self._teacher_network.state_dict(), self._save_path)
            
            self._scheduler.step()
        
        logging.info("Training concluded (best metric={:.4f} at epoch={})".format(min_metric, best_metric_epoch))
    
    def train_epoch(self, device):
        """ Train one epoch with mean-teacher style setup
        
        :param device: device to use
        """
        
        self._student_network.to(device)
        self._teacher_network.to(device)
        self._student_network.train()
        self._teacher_network.train()
        self._ema.to(device)
        
        losses = []
        for item in tqdm.tqdm(self._train_dataloader):
            # For mean-teacher style loaders, we have 4 images
            B = item[0].shape[0]
            student_imageA = item[0].to(device)
            student_imageB = item[1].to(device)
            teacher_imageA = item[2].to(device)
            teacher_imageB = item[3].to(device)
            label = item[-1].to(device)
            
            self._optimizer.zero_grad()
            
            # Get student predictions
            student_predictions = self._student_network(
                torch.stack([student_imageA, student_imageB], dim=1).flatten(0,1)
            )
            student_predictions = {k: student_predictions[k].unflatten(0, (B,2)) for k in student_predictions.keys()}
            
            # Get teacher predictions
            teacher_predictions = self._teacher_network(
                torch.stack([teacher_imageA, teacher_imageB], dim=1).flatten(0,1)
            )
            teacher_predictions = {k: teacher_predictions[k].unflatten(0, (B,2)) for k in teacher_predictions.keys()}
            
            step_loss = self._loss_function(student_predictions=student_predictions,
                                            teacher_predictions=teacher_predictions,
                                            targets=label)
            step_loss.backward()
            self._optimizer.step()
            
            # do EMA update for mean teacher
            self._ema.update(self._student_network.parameters())
            self._ema.copy_to(self._teacher_network.parameters())
            losses.append(step_loss.detach().cpu().item())
        
        return np.array(losses).mean()
