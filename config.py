import yaml

def load_config(config_file_path):
    with open(config_file_path, "r") as config:
        return yaml.safe_load(config)

class ConfigLoader:
    def __init__(self, config_path):
        self.config = load_config(config_path)
    
    def traing_percentage(self):
        return self.config["training"]["train_percentage"]

    def batch_size(self):
        return self.config["training"]["batch_size"]

    def resolution(self):
        return self.config["images"]["resolution"]

    def num_load_workers(self):
        return self.config["loader"]["workers"]

    def learning_rate(self):
        return float(self.config["training"]["learning_rate"])

    def epochs(self):
        return self.config["training"]["epochs"]

    def manual_seed(self):
        return self.config["training"]["seed"]

if __name__ == "__main__":
    print(load_config("config.yaml")["training"]["batch_size"])