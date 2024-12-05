import csv

with open("config.yaml", "r") as file:
      config = yaml.safe_load(file)
      # Training
      init_x = config["pos"]["init_x"]
      init_y = config["pos"]["init_y"]
      goal_x = config["pos"]["goal_x"]
      goal_y = config["pos"]["goal_y"]
      goal_y = config["num_episodes"]
