#!/usr/bin/env python3
import functools
import os

from baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_custom_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack

from stable_baselines.common.vec_env import VecVideoRecorder
import wandb

def train(*, env_id, num_env, hps, num_timesteps, seed):
    venv = VecFrameStack(
    		       make_custom_env(env_id,
                            num_env,
                            seed,
                            start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                            max_episode_steps=hps.pop('max_episode_steps'),
                            record_when = hps.pop("record_when"),
                            size = hps.pop("size"),
                            random_actions = hps.pop("random_actions"),
                            add_noise = hps.pop("add_noise"),
                            record_coverage = hps.pop('record_coverage'),
                            exp_name = hps.pop('exp_name')
                            ),
    hps.pop('frame_stack'))

#    venv.num_envs = num_env
   # VecVideoRecorder(venv,"./video", record_video_trigger = lambda episode_id: episode_id%500)
    # venv.score_multiple = {'Mario': 500,
    #                        'MontezumaRevengeNoFrameskip-v4': 100,
    #                        'GravitarNoFrameskip-v4': 250,
    #                        'PrivateEyeNoFrameskip-v4': 500,
    #                        'SolarisNoFrameskip-v4': None,
    #                        'VentureNoFrameskip-v4': 200,
    #                        'PitfallNoFrameskip-v4': 100,
    #                        }[env_id]
    venv.score_multiple = 1
    
    
    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')
    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]
    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
                scope='pol',
                ob_space=ob_space,
                ac_space=ac_space,
                update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
                proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
                dynamics_bonus = hps.pop("dynamics_bonus")
            ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=hps.pop('nminibatches'),
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
    )
    agent.start_interaction([venv])
    if hps.pop('update_ob_stats_from_random_agent'):
        agent.collect_random_statistics(num_timesteps=128*50)
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

    counter = 0
    while True:
        info = agent.step()
        if info['update']:
            logger.logkvs(info['update'])
            logger.dumpkvs()
            counter += 1
        if agent.I.stats['tcount'] > num_timesteps:
            break

    agent.stop_interaction()


def add_env_params(parser):
    parser.add_argument('--env', help='environment ID', default='MiniGrid-DoorKey-8x8-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    parser = arg_parser()
    add_env_params(parser)
    # Short runs
    parser.add_argument('--num-timesteps', type=int, default=int(1000448))    
    # Middle runs
    # parser.add_argument('--num-timesteps', type=int, default=int(2000000))
    # Long runs
    # parser.add_argument('--num-timesteps', type=int, default=int(10000000))
    
    parser.add_argument('--exp_name', type=str, default='Just another test')
    parser.add_argument('--num_env', type=int, default=8)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_ext', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='cnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=1.)
    parser.add_argument('--dynamics_bonus', type=int, default=0)
    parser.add_argument('--number_stack', type=int, default=1)
    
    parser.add_argument('--record_when', type=int, default=400)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--random_actions', default=False)
    parser.add_argument('--record_coverage', default=False)
    parser.add_argument('--add_noise', default=False)


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger.configure(dir=logger.get_dir(), format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
            f.write(args.tag)
        # shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(logger.get_dir(), 'code'))

    mpi_util.setup_mpi_gpus()

    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    hps = dict(
        frame_stack=args.number_stack,
        nminibatches=8,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        dynamics_bonus = args.dynamics_bonus,
        record_coverage = args.record_coverage,
        record_when = args.record_when,
        size = args.size,
        random_actions = args.random_actions,
        add_noise = args.add_noise,
        exp_name = args.exp_name
        
    )
    wandb.init(project="thesis", group = "Random_Network_Distillation", entity = "lukischueler", name = args.exp_name, config = hps)
    wandb.config.update(args)
    
    # Define the custom x axis metric
    wandb.define_metric("Number of Episodes")
    wandb.define_metric("Frames seen")
    wandb.define_metric("Number of Updates")

    # Define which metrics to plot against that x-axis
    wandb.define_metric("Episode Reward", step_metric='Number of Updates')
    wandb.define_metric("Length of Episode", step_metric='Number of Updates')
    wandb.define_metric("Recent Best Reward", step_metric='Number of Updates')
    wandb.define_metric("Intrinsic Reward (Batch)", step_metric='Number of Updates')
    wandb.define_metric("Extrinsic Reward (Batch)", step_metric='Number of Updates')
    
    wandb.define_metric("Episode Reward", step_metric='Frames seen')
    wandb.define_metric("Length of Episode", step_metric='Frames seen')
    wandb.define_metric("Recent Best Reward", step_metric='Frames seen')
    
    wandb.define_metric("Intrinsic Reward (Batch)", step_metric='Frames seen')
    wandb.define_metric("Extrinsic Reward (Batch)", step_metric='Frames seen')
    
    
    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
        num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
    main()
