import yaml

with open("config.yaml", "r") as file:
      config = yaml.safe_load(file)
      init_x = config["pos"]["init_x"]
      init_y = config["pos"]["init_y"]
      goal_x = config["pos"]["goal_x"]
      goal_y = config["pos"]["goal_y"]
      num_episodes = config["num_episodes"]
