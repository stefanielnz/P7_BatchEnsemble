#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J Test_Python
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify memory requirements (e.g., 2GB per core) --
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set email address for notifications --
#BSUB -u s242107@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file --
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err
echo "loading module"
module load python3/3.12.4
echo "done loading module"

nvidia-smi

source .venv/bin/activate
### Execute the Python script
echo "environment has been activated"

pip install ivon-opt

python --version
python Complex_Test.py # email und file anpassen, in terminal "bsub < submit_test.sh" und status zu bekommen "bstat"