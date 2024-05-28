#!/usr/bin/env bash
# Submit with: `qsub -v WANDB_API_KEY=<API_KEY>,ADD_ARGS='world_model=gpt2_pretrained_frozen ...' metacentrum_job.sh`
### Define machine requirements ###
#PBS -N daisy-iris
#PBS -q gpu_long
#PBS -l select=1:ncpus=4:mem=40gb:scratch_local=100gb:ngpus=1:gpu_mem=38gb
#PBS -l walltime=336:00:00
#PBS -m ae

HOME_DIR=/storage/plzen1/home/patrikvacal

### Check for Weights and Biases API key ###
if test -z $WANDB_API_KEY
then
  echo "WANDB_API_KEY not found! Exiting."
  exit
fi


### Prepare RUN env ###
module add conda-modules  # Newest conda version
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# pip install -r requirements.txt

# Copy the environment to the scratch directory,
#  otherwise "Stale file handle" error may occur when more jobs starts at the same time.
mkdir -p $SCRATCHDIR/envs/conda
cp -r $HOME_DIR/envs/conda/iris $SCRATCHDIR/envs/conda
conda activate $SCRATCHDIR/envs/conda/iris

export WANDB__SERVICE_WAIT=300

### Define cpu LIMITS for the script! ###
export OMP_NUM_THREADS=$PBS_NUM_PPN
export OPENBLAS_NUM_THREADS=$PBS_NUM_PPN
export MKL_NUM_THREADS=$PBS_NUM_PPN
export VECLIB_MAXIMUM_THREADS=$PBS_NUM_PPN
export NUMEXPR_NUM_THREADS=$PBS_NUM_PPN

### Cache dirs ###
export TMPDIR=$SCRATCHDIR/tmp
export WANDB_CACHE_DIR=$SCRATCHDIR/cache_wandb
export TORCH_HOME=$SCRATCHDIR/cache_pytorch
export TORCH_HUB=$SCRATCHDIR/cache_pytorch

### Copy results back from scratch even if script fails ###
on_exit() {
  cd $SCRATCHDIR
  tar czvf job-$PBS_JOBID.tgz iris_default
  cp job-$PBS_JOBID.tgz $HOME_DIR/job_results  || export CLEAN_SCRATCH=false
}
trap on_exit TERM EXIT

### RUN ###
cp -r $HOME_DIR/jobs/iris_default $SCRATCHDIR
cd $SCRATCHDIR/iris_default
echo "Running main.py with ADD_ARGS: $ADD_ARGS"
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=online $ADD_ARGS
