#!/bin/bash
#SBATCH --job-name=ESRNET2        # create a short name for your job
#SBATCH --partition=ais-gpu 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32         
#SBATCH --mem-per-cpu=12G
#SBATCH --time=7-00:00:00       # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email when job fails
#SBATCH --mail-user=m.aleshin@skoltech.ru
#SBATCH --output=experiments/slurm_logs/esrnet_%x_%j.txt 

source /beegfs/home/m.aleshin/.bashrc
conda activate esrgan

cd /beegfs/home/m.aleshin/projects/superresolution/Real-ESRGAN

# export USE_SIMPLE_THREADED_LEVEL3=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# nvidia-smi

export OMP_NUM_THREADS=2
# 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=13112 \
    realesrgan/train.py -opt options/train_realesrnet_x2plus_mod.yml \
                        --launcher pytorch --auto_resume #--debug

# 2
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=13110 \
#     realesrgan/train.py -opt options/train_realesrgan_x4plus_satellite.yml \
#                         --launcher pytorch --auto_resume

# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=13131 \
#     realesrgan/train.py -opt options/finetune_realesrgan_x4plus_satellite.yml \
#                         --launcher pytorch --auto_resume
# python realesrgan/train.py -opt options/train_realesrnet_x4plus_satellite.yml --debug