import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import utils 
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
from utils import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO
from stable_baselines3 import DQN
CHECKPOINT_DIR = "trains/train1/"
print(CHECKPOINT_DIR)
LOG_DIR = "./logs/"
    
env = gym_super_mario_bros.make("SuperMarioBros-v0") # Generates the environment
env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits the joypads moves with important moves
#utils.startGameRand(env)
env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
env = DummyVecEnv([lambda: env])
# Alternatively, you may use SubprocVecEnv for multiple CPU processors
env = VecFrameStack(env, 4, channels_order="last") # Stack frames
env = VecMonitor(env, CHECKPOINT_DIR+"TestMonitor") # Monitor your progress

model = PPO.load(CHECKPOINT_DIR+"best_model")
utils.startGameModel(env, model)