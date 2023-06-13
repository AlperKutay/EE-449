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
for i in range (3,6):
    CHECKPOINT_DIR = "./train"+str(i)+"/"
    print(CHECKPOINT_DIR)
    LOG_DIR = "./logs/"
    # Start the environment
    # Set the device to use

    env = gym_super_mario_bros.make("SuperMarioBros-v0") # Generates the environment
    env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits the joypads moves with important moves
    #utils.startGameRand(env)

    env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
    env = DummyVecEnv([lambda: env])
    # Alternatively, you may use SubprocVecEnv for multiple CPU processors
    env = VecFrameStack(env, 4, channels_order="last") # Stack frames
    env = VecMonitor(env, CHECKPOINT_DIR+"TestMonitor") # Monitor your progress


    callback = SaveOnBestTrainingRewardCallback(save_freq=100000, check_freq=1000,chk_dir=CHECKPOINT_DIR)
    if(i == 3):
        model = DQN("CnnPolicy",
                    env,
                    batch_size=192,
                    verbose=1,
                    learning_starts=10000,
                    learning_rate=0.001,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    train_freq=8,
                    buffer_size=10000,
                    tensorboard_log=LOG_DIR)
    elif( i == 4):
        model = DQN("CnnPolicy",
                    env,
                    batch_size=216,
                    verbose=1,
                    learning_starts=5000,
                    learning_rate=5e-3,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    train_freq=8,
                    buffer_size=12000,
                    tensorboard_log=LOG_DIR)
    elif(i == 5):
        model = DQN("MlpPolicy",
                    env,
                    batch_size=192,
                    verbose=1,
                    learning_starts=10000,
                    learning_rate=8e-3,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    train_freq=8,
                    buffer_size=10000,
                    tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=1000000, log_interval=1, callback=callback)

"""
env = gym_super_mario_bros.make("SuperMarioBros-v0") # Generates the environment
env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits the joypads moves with important moves
#utils.startGameRand(env)
env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
env = DummyVecEnv([lambda: env])
# Alternatively, you may use SubprocVecEnv for multiple CPU processors
env = VecFrameStack(env, 4, channels_order="last") # Stack frames
env = VecMonitor(env, CHECKPOINT_DIR+"TestMonitor") # Monitor your progress

model = PPO.load("./train/best_model.zip")
utils.startGameModel(env, model)


model = DQN("CnnPolicy",
                    env,
                    batch_size=192,
                    verbose=1,
                    learning_starts=10000,
                    learning_rate=5e-3,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    train_freq=8,
                    buffer_size=10000,
                    tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1, callback=callback)
"""
