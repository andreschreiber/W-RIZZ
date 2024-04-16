import json
from pathlib import Path


class Experiment:
    def __init__(self, path, should_exist=False):
        """ Constructor for experiment 
        
        :param path: path to experiment folder
        :param should_exist: whether the folder should exist or not
        """
        if should_exist and not path.exists():
            raise FileNotFoundError("Experiment directory does not exist but should")
        elif not should_exist and path.exists():
            raise FileNotFoundError("Experiment directory exists but should not")
        
        path.mkdir(parents=True, exist_ok=True)
        self.folder = path
        self.model_file = path / Path('model.pth')
        self.train_log_file = path / Path('train_log.txt')
        self.test_log_file = path / Path('test_log.txt')
        self.results_file = path / Path('results.json')
        self.settings_file = path / Path('settings.json')

    def read_settings(self):
        """ Reads JSON file for experiment settings """
        settings = None
        with open(self.settings_file, 'r') as f:
            settings = json.load(f)
        return settings
    
    def write_settings(self, settings):
        """ Writes JSON file for experiment settings """
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
