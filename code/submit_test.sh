#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J Test_Python
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify memory requirements (e.g., 2GB per core) --
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
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

### Execute the Python script
python Test.py # email und file anpassen, in terminal "bsub < submit_test.sh" und status zu bekommen "bstat"
