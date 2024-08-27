#!/bin/bash
#SBATCH --job-name=ESRGAN       # create a short name for your job
#SBATCH --partition=ais-gpu 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus=4                 # 4
#SBATCH --cpus-per-task=64       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6-00:00:00        # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email when job fails
#SBATCH --mail-user=m.aleshin@skoltech.ru
#SBATCH --output=experiments/slurm_logs/esrgan_%x_%j.txt   

source /beegfs/home/m.aleshin/.bashrc
conda activate esrgan

cd /beegfs/home/m.aleshin/projects/superresolution/Real-ESRGAN

export OMP_NUM_THREADS=8

# nvidia-smi

# 1
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=13110 \
#     realesrgan/train.py -opt options/train_realesrnet_x4plus_satellite.yml \
#                         --launcher pytorch --auto_resume

# 2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=13110 \
    realesrgan/train.py -opt options/train_realesrgan_x4plus_satellite.yml \
                        --launcher pytorch --auto_resume

# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=13131 \
#     realesrgan/train.py -opt options/finetune_realesrgan_x4plus_satellite.yml \
#                         --launcher pytorch --auto_resume
# # python realesrgan/train.py -opt options/train_realesrnet_x4plus_satellite.yml --debug