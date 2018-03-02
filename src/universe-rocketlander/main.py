import json
from GameEnv import GameEnv
game_env = GameEnv(json.load(open("config.json", "r")))
game_env.run()
