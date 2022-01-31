#! /bin/sh

pip list --format=freeze > requirements.txt
conda env export > conda_env.yml
conda clean --all -y

