from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from deepq import learn
import re
import argparse
import datetime
import os

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)


def wrap_atari_evaluate_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--learning-starts', type=int, default=50000)
    parser.add_argument('--target-network-update-freq', type=int, default=10000)
    parser.add_argument('--exploration-fraction', type=float, default=0.05)
    parser.add_argument('--exploration-final-eps', type=float, default=0.1)
    parser.add_argument('--checkpoint-freq', type=int, default=int(1e5))
    parser.add_argument('--double-q', action='store_true', default=False)
    parser.add_argument('--num-ensemble', type=int, default=10)               # number of ensemble is 10
    parser.add_argument('--gradient-norm', type=bool, default=True)           # gradient norm is True
    # action selection parameters
    parser.add_argument('--action-selection', type=str, default="sample", choices=["sample", "vote", "ucb", "ids"])   # different action selection methods
    # bonus parameters
    parser.add_argument('--reward-type', type=str, default="none", choices=["none", "ucb"])  # different kinds of intrinsic reward
    parser.add_argument('--rew-immed-ratio', type=float, default=0.001)
    parser.add_argument('--rew-nextq-ratio', type=float, default=0.001)         # ratio of BDQN with intrinsic reward
    parser.add_argument('--normrew', type=bool, default=False)                  # if norm the reward
    parser.add_argument('--normnxq', type=bool, default=True)                   # if norm the next-q value
    parser.add_argument('--rew-immed-ratio-ebu', type=float, default=0.00005)
    parser.add_argument('--rew-nextq-ratio-ebu', type=float, default=0.00005)   # ratio of BEBU with intrinsic reward
    parser.add_argument('--normrew-ebu', type=bool, default=False)            # if norm the reward
    parser.add_argument('--normnxq-ebu', type=bool, default=True)             # if norm the next-q value
    # random prior parameters
    parser.add_argument('--prior', action='store_true', default=False)        # if use randomized prior function
    parser.add_argument('--prior-scale', type=float, default=3.0)             # ratio of randomized prior function
    # if activate EBU
    parser.add_argument('--ebu', action='store_true', default=False)          # ratio of randomized prior function
    parser.add_argument('--beta', type=float, default=0.5)                    # beta of EBU
    parser.add_argument('--max-episode-steps', type=int, default=None)
    args = parser.parse_args()

    # log
    log_dir = os.path.join("result", re.sub("NoFrameskip-v4", "", args.env))
    if args.ebu:
        log_dir += "-BEBU"
    else:
        log_dir += "-BDQN"

    if args.action_selection != 'sample':
        log_dir += "-action-" + str(args.action_selection)
    if args.reward_type != 'none':
        log_dir += "-reward-" + str(args.reward_type)
        if args.ebu:
            if args.rew_immed_ratio_ebu > 0:
                log_dir += ("-" + str(args.rew_immed_ratio_ebu))
                if args.normrew_ebu:
                    log_dir += "-normrew"
            if args.rew_nextq_ratio_ebu > 0:
                log_dir += ("-" + str(args.rew_nextq_ratio_ebu))
                if args.normnxq_ebu:
                    log_dir += "-normnxq"
        else:
            if args.rew_immed_ratio > 0:
                log_dir += ("-" + str(args.rew_immed_ratio))
                if args.normrew:
                    log_dir += "-normrew"
            if args.rew_nextq_ratio > 0:
                log_dir += ("-" + str(args.rew_nextq_ratio))
                if args.normnxq:
                    log_dir += "-normnxq"
    if args.prior:
        log_dir += "-prior-" + str(args.prior)
    log_dir += datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")

    logger.configure(dir=log_dir)
    with open(os.path.join(log_dir, 'parameters.txt'), 'w') as f:
        f.write("\n".join([str(x[0]) + ": " + str(x[1]) for x in vars(args).items()]))

    max_episode_steps = args.max_episode_steps
    if args.ebu and args.env in ['SeaquestNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4',
                    'MontezumaRevengeNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'BattleZoneNoFrameskip-v4']:
        max_episode_steps = 4500

    env = make_atari(args.env, max_episode_steps=max_episode_steps)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    # env for evaluation
    if not os.path.exists('tmp'):
        os.mkdir("tmp")
    eval_env = make_atari(args.env, max_episode_steps=max_episode_steps)
    eval_env = bench.Monitor(eval_env, "tmp/"+datetime.datetime.now().strftime("%m-%d-%H-%M-%S"), allow_early_resets=True)
    eval_env = wrap_atari_dqn(eval_env)

    model = learn(
        env,
        eval_env,
        ebu=args.ebu,
        beta=args.beta,
        action_selection=args.action_selection,       # action selection parameters
        reward_type=args.reward_type,
        rew_immed_ratio=args.rew_immed_ratio,
        rew_nextq_ratio=args.rew_nextq_ratio,
        normrew=args.normrew,
        normnxq=args.normnxq,                         # intrinsic reward parameters (DQN)
        rew_immed_ratio_ebu=args.rew_immed_ratio_ebu,
        rew_nextq_ratio_ebu=args.rew_nextq_ratio_ebu,
        normrew_ebu=args.normrew_ebu,
        normnxq_ebu=args.normnxq_ebu,                 # intrinsic reward parameters (EBU)
        gradient_norm=args.gradient_norm,
        num_ensemble=args.num_ensemble,
        prior=args.prior,
        prior_scale=args.prior_scale,
        double_q=args.double_q,
        param_noise=False,
        train_freq=4,
        gamma=0.99,
        lr=args.lr,
        total_timesteps=args.num_timesteps,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        learning_starts=args.learning_starts,
        target_network_update_freq=args.target_network_update_freq,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=log_dir)

    model.q_network.save_weights(os.path.join(log_dir, 'model_20M.h5'))
    env.close()


if __name__ == '__main__':
    main()
