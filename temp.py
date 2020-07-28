import config
import yaml

args = config.config()
d = vars(args)
with open("config.yml", "w") as f:
    yaml.dump(args, f)
