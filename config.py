import yaml




def load_config(config_file_path):
    with open(config_file_path, "r") as config:
        return yaml.safe_load(config)






class ConfigLoader:
    def __init__(self, config_path):
        self.config = load_config(config_path)
    
    def traing_percentage(self):
        return self.config["training"]["train_percentage"]

if __name__ == "__main__":
    print(load_config("config.yaml")["training"]["batch_size"])