# World Models in Reinforcement Learning - code repository

- Code for my bachelor thesis **World Models in Reinforcement Learning**.
- This codebase is built on top of IRIS world model (https://github.com/eloialonso/iris).

## Implemented experiments

List of all the experiments we implemented in this codebase on top of IRIS world model.
Only the most promising of them were used in the thesis because of limited time and resources.

- Using pretrained GPT2 transformer model as the world model.
- Freezing transformer world model layers.
- Loading and fixing pretrained models VQ-VAE model for the world model.
- Neural Control Policies (NCP) world model with single-step predictions of all next state tokens.
- Neural Control Policies (NCP) world model with multi-step autoregressive predictions of next state tokens.
- Different losses for VQ-VAE model.
- Adding noise to the observations to see how world models deal with it.

## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Launch a training run

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

### Train Locally

1. Login to Weights & Biases in terminal with `wandb login`
2. Run the training script with desired configuration parameters (PARAM=VALUE):
```bash
python src/main.py env.train.id=<ENV_NAME> <OTHER_CONFIG_PARAMETERS in format PARAM=VALUE>
```

### Train in MetaCentrum

1. Set you Weights & Biases API key in the `.env` file:
   - Copy `.env.template` to `.env`
   - Fill in the `WANDB_API_KEY`
2. Submit the job to MetaCentrum with the following command:
```bash
bash metacentrum_job.sh env.train.id=<ENV_NAME> <OTHER_CONFIG_PARAMETERS in format PARAM=VALUE>
```

## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files or set the parameters in the command line when launching the training script.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Project structure

Here we list the most important folders and files in the project structure.

- `config/`: contains the configuration files.
- `scripts/`: contains scripts for evaluating, resuming and playing with the trained agent.
  - `eval.py`: Launch `python ./scripts/eval.py` to evaluate the run.
  - `resume.sh`: Launch `./scripts/resume.sh` to resume a training that crashed.
  - `play.sh`: Tool to visualize some interesting aspects of the run.
    - Launch `./scripts/play.sh` to watch the agent play live in the environment. If you add the flag `-r`, the left panel displays the original frame, the center panel displays the same frame downscaled to the input resolution of the discrete autoencoder, and the right panel shows the output of the autoencoder (what the agent actually sees).
    - Launch `./scripts/play.sh -w` to unroll live trajectories with your keyboard inputs (i.e. to play in the world model). Note that for faster interaction, the memory of the Transformer is flushed every 20 frames.
    - Launch `./scripts/play.sh -a` to watch the agent play live in the world model. Note that for faster interaction, the memory of the Transformer is flushed every 20 frames.
    - Launch `./scripts/play.sh -e` to visualize the episodes contained in `media/episodes`.
    - Add the flag `-h` to display a header with additional information.
    - Press '`,`' to start and stop recording. The corresponding segment is saved in `media/recordings` in mp4 and numpy formats.
    - Add the flag `-s` to enter 'save mode', where the user is prompted to save trajectories upon completion.
- `src/`: contains the experiment code.
  - `envs/`: contains wrappers and utilities for working with the Atari environment.
  - `game/`: contains tools for user and agent interaction with the environment.
  - `models/`: contains implementation of agent and world model architectures.
    - `tokenizer/`: contains the VQ_VAE tokenizer implementation.
    - `actor_critic.py`: contains the implementation of the actor-critic agent.
    - `slicer.py`: contains the implementation of the Slicer, Head and Embedder classes for the world models implementations.
    - `world_model_transformer.py`, `kv_caching.py`: contains the implementation of the transformer world model.
    - `world_model_ncp_single_step.py`: contains our implementation of the NCP world model.
    - `world_model_ncp_multi_step.py`: contains our implementation of the NCP world model with autoregressive predictions. Not used in the experiments.
    - `world_model_dummy.py`: contains our dummy world model, which uses the real environment. Not used in the experiments.
  - `trainer.py`: Responsible for setup and training of the agent and the world model in tandem.
  - `agent.py`: Implementation of the Agent.
  - `collectory.py`: Responsible for collecting data from the environment and storing it in the replay buffer.
- `runs/`: contains the results of the runs.
  - Each run is stored in a separate folder with the name of the run, which is generated automatically based on the current date and time.
- `.env.template`: template for the environment variables, copy it to `.env` and fill in the values before running the code in MetaCentrum.
- `metacentrum_job.sh`: bash script to run the code in MetaCentrum.
- `requirements.txt`: contains the Python dependencies required to run the code.

## Run folder

For each run, a folder is created in `runs/` with the following structure

- `config`: copy of the configuration files used for the run.
- `src`: copy of the source code used for the run.
- `checkpoints`: contains the last checkpoint of the model, its optimizer and the dataset.
- `media`:
  - `episodes`: contains train / test / imagination episodes for visualization purposes.
  - `reconstructions`: contains original frames alongside their reconstructions with the autoencoder.
- `scripts`: **from the run folder**, you can use the following three scripts.

## Credits

- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/eloialonso/iris](https://github.com/eloialonso/iris)
