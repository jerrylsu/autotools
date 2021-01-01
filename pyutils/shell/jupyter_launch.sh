#!/bin/sh

# enter docker
cd ~
jupyter notebook --generate-config
echo 'Please input jupyter notebook password'
jupyter notebook password
echo 'Password input success!'

chmod 777 ~/.jupyter/jupyter_notebook_config.json
PASSWORD=$(cat ~/.jupyter/jupyter_notebook_config.json | grep password | awk -F '"' '{print $4}')
echo "c.NotebookApp.ip='*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.password = '$PASSWORD'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8480" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.notebook_dir = '/home'" >> ~/.jupyter/jupyter_notebook_config.py

nohup jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port 8180 > jupyter.log 2>&1 &

conda install nb_conda  # notebook change virtual environment
