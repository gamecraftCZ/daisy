from functools import partial 
from pathlib import Path

import hydra
import numpy as np
from gym.wrappers import TransformObservation
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Game
from models.actor_critic import ActorCritic
from models.world_model_dummy import WorldModelDummy
from models.world_model_ncp_multiple_step import WorldModelNcpMultipleStep
from models.world_model_ncp_single_step import WorldModelNcpSingleStep
from models.world_model_transformer import WorldModelTransformer


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    def env_fn(*args, **kwargs):
        cfg_env = {**cfg.env.test}
        del cfg_env["noise_std"]
        env = partial(instantiate, config=cfg_env)(*args, **kwargs)
        env = TransformObservation(env, lambda obs: np.clip(
            (obs + np.random.randn(*obs.shape) * 255. * cfg.env.test.noise_std)
            , 0, 255.)
                                   ) if cfg.env.test.noise_std else env
        return env
    test_env = SingleProcessEnv(env_fn)

    if cfg.mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    
    if cfg.mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    else:
        tokenizer = instantiate(cfg.tokenizer)

        if cfg.world_model.type == 'transformer':  # Transformer based world model
            world_model = WorldModelTransformer(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(cfg.world_model.transformer))

        elif cfg.world_model.type == 'dummy':  # Real environment is used instead of world model
            world_model = WorldModelDummy(tokenizer=tokenizer, obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=cfg.world_model.dummy_config, env_count=cfg.training.world_model.batch_num_samples, device=device)

        elif cfg.world_model.type == "ncp_single_step":  # NCP based world model
            world_model = WorldModelNcpSingleStep(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(cfg.world_model.ncp_single_step))

        elif cfg.world_model.type == "ncp_multiple_step":  # NCP based world model with multiple steps as the transformer model
            world_model = WorldModelNcpMultipleStep(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(cfg.world_model.ncp_multiple_step))

        else:
            raise NotImplementedError("Unknown world model type: {}".format(cfg.world_model.type))

        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
        agent = Agent(tokenizer, world_model, actor_critic).to(device)
        agent.load(Path('checkpoints/last.pt'), device)

        if cfg.mode == 'play_in_world_model':
            env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn())
            keymap = cfg.env.keymap
        
        elif cfg.mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=cfg.reconstruction)
            keymap = 'empty'
            if cfg.reconstruction:
                size[1] *= 3

        elif cfg.mode == 'agent_in_world_model':
            wm_env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn())
            env = AgentEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = 'empty'

    game = Game(env, keymap_name=keymap, size=size, fps=cfg.fps, verbose=bool(cfg.header), record_mode=bool(cfg.save_mode))
    game.run()


if __name__ == "__main__":
    main()
