import yaml

def load_config(config_file_path):
    with open(config_file_path, "r") as config:
        return yaml.safe_load(config)

class ConfigLoader:
    def __init__(self, config_path):
        self.config = load_config(config_path)
    
    def train_percentage(self) -> float:
        return self.config["training"]["train_percentage"]

    def test_percentage(self) -> float:
        return self.config["training"]["test_percentage"]

    def validation_percentage(self) -> float:
        return self.config["training"]["validation_percentage"]

    def batch_size(self) -> int:
        return self.config["training"]["batch_size"]

    def resolution(self) -> int:
        return self.config["images"]["resolution"]

    def num_load_workers(self) -> int:
        return self.config["loader"]["workers"]

    def learning_rate(self) -> float:
        return float(self.config["training"]["learning_rate"])

    def epochs(self) -> int:
        return self.config["training"]["epochs"]

    def manual_seed(self) -> int:
        return self.config["training"]["seed"]

if __name__ == "__main__":
    print(load_config("config.yaml")["training"]["batch_size"])