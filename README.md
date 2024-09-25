# Experiments on IRIS world model

- Project by Patrik Vácal.
- This repository contains experiments built on top of IRIS world model.
- Original code from https://github.com/eloialonso/iris

## Implemented experiments on top of IRIS
- Pretrained transformer model as the world model.
- Pretrained VQ-VAE model for the world model.
- Different architectures for the world model.
- Different losses for VQ-VAE model.

## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Launch a training run

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=online
```

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files directly.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Run folder

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   last.pt
|   |   optimizer.pt
|   |   ...
│   │
│   └─── dataset
│       │   0.pt
│       │   1.pt
│       │   ...
│
└─── config
│   |   trainer.yaml
|
└─── media
│   │
│   └─── episodes
│   |   │   ...
│   │
│   └─── reconstructions
│   |   │   ...
│
└─── scripts
|   |   eval.py
│   │   play.sh
│   │   resume.sh
|   |   ...
|
└─── src
|   |   ...
|
└─── wandb
    |   ...
```

- `checkpoints`: contains the last checkpoint of the model, its optimizer and the dataset.
- `media`:
  - `episodes`: contains train / test / imagination episodes for visualization purposes.
  - `reconstructions`: contains original frames alongside their reconstructions with the autoencoder.
- `scripts`: **from the run folder**, you can use the following three scripts.
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

## Credits

- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/eloialonso/iris](https://github.com/eloialonso/iris)
