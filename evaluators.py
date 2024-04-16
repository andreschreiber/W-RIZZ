import tqdm
import torch


class Evaluator:
    """ Evaluator base class """
    def __init__(self, dataloader, metrics):
        """ Constructor for evaluator 
        
        :param dataloader: dataloader to use for evaluation
        :param metrics: metrics to use (dictionary of form, name: metric_instance)
        """
        self._dataloader = dataloader
        self._metrics = metrics
    
    def validate(self, network, device):
        raise NotImplementedError
    
    
class BasicEvaluator(Evaluator):
    def __init__(self, dataloader, metrics):
        """ Constructor for evaluator 
        
        :param dataloader: dataloader to use for evaluation
        :param metrics: metrics to use (dictionary of form, name: metric_instance)
        """
        super().__init__(dataloader, metrics)
    
    def validate(self, network, device):
        """ Performs evaluation
        
        :param network: network to use
        :param device: device to use
        :returns: dictionary of results (where key is metric name and value is metric value)
        """
        # Reset all metrics
        for k in self._metrics.keys():
            self._metrics[k].reset()
        
        network.to(device)
        network.eval()
        # Iterate through dataset and evaluate
        with torch.no_grad():
            for item in tqdm.tqdm(self._dataloader):
                B = item[0].shape[0]
                imageA = item[0].to(device)
                imageB = item[1].to(device)
                label = item[-1].to(device)
                output = network(torch.stack([imageA, imageB], dim=1).flatten(0,1))
                predictions = output['prediction'].unflatten(0, (B,2))
                for k in self._metrics.keys():
                    # Calculate metric--this updates the metric object
                    result = self._metrics[k](predictions, label)
        return {k: self._metrics[k].compute().cpu().item() for k in self._metrics.keys()}
