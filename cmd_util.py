"""
Helpers for scripts like run_atari.py.
"""

import os

import gym
from gym.wrappers import FlattenDictWrapper
from gym.wrappers import Monitor as VideoMonitor
from mpi4py import MPI
from baselines import logger
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from monitor import Monitor
from atari_wrappers import make_atari, wrap_deepmind
from vec_env import SubprocVecEnv
import gym_minigrid

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper, VectorizedWrapper



def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500):
    """
    Create a wrapped, monitored SubprocVecEnv for the MiniGrid-Environment
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            # env._max_episode_steps = max_episode_steps*4
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)

            # env = VideoMonitor(env, "./RNDvideo", video_callable = lambda episode_id: episode_id%100000, force =True)
           # env = VecVideoRecorder(env,"./video", record_video_trigger = lambda episode_id: episode_id%500)
            return ImgObsWrapper(RGBImgPartialObsWrapper(env))
        return _thunk
    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i+start_index) for i in range(num_env)])



def make_atari(env_id, max_episode_steps=4500):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps*4
    assert 'NoFrameskip' in env.spec.id
    env = StickyActionEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if "Montezuma" in env_id or "Pitfall" in env_id:
        env = MontezumaInfoWrapper(env, room_address=3 if "Montezuma" in env_id else 1)
    else:
        env = DummyMontezumaInfoWrapper(env)
    env = AddRandomStateToInfo(env)
    return env



def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser
