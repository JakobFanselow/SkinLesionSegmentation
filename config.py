import yaml

def load_config(config_file_path):
    with open(config_file_path, "r") as config:
        return yaml.safe_load(config)

class ConfigLoader:
    def __init__(self, wandb_config):
        self.config = wandb_config

    def _get_val(self, category, key):
        if key in self.config:
            return self.config[key]
        return self.config[category][key]

    def model(self) -> str:
        return str(self._get_val("model", "model"))
     
    def train_percentage(self) -> float:
        return float(self._get_val("training", "train_percentage"))

    def test_percentage(self) -> float:
        return float(self._get_val("training", "test_percentage"))

    def validation_percentage(self) -> float:
        return float(self._get_val("training", "validation_percentage"))

    def batch_size(self) -> int:
        return int(self._get_val("training", "batch_size"))

    def resolution(self) -> int:
        return int(self._get_val("images", "resolution"))

    def num_load_workers(self) -> int:
        return int(self._get_val("loader", "workers"))

    def max_learning_rate(self) -> float:
        return float(self._get_val("training", "max_learning_rate"))

    def learning_rate(self) -> float:
        return float(self._get_val("training", "learning_rate"))

    def epochs(self) -> int:
        return int(self._get_val("training", "epochs"))

    def manual_seed(self) -> int:
        return int(self._get_val("training", "seed"))

    def weight_decay(self) -> float:
        return float(self._get_val("training", "weight_decay"))

    def drop_last(self) -> bool:
        return bool(self._get_val("training", "drop_last"))

    def max_norm(self) -> float:
        return float(self._get_val("training", "max_norm"))
    
    def dice_weight(self) -> float:
        return float(self._get_val("loss", "dice_weight"))
    
    def bce_weight(self) -> float:
        return 1 - float(self._get_val("loss", "dice_weight"))

    def kernel_size(self) -> float:
        try:
            return int(self._get_val("model","kernel_size"))
        except:
            return None

    
    def exclude_bottleneck(self) -> float:
        try:
            return bool(self._get_val("model","exclude_bottleneck"))
        except:
            return None

if __name__ == "__main__":
    print(load_config("config.yaml")["training"]["batch_size"])