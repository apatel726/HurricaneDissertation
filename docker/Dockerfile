FROM tensorflow/tensorflow:1.14.0-py3-jupyter
COPY files/ /tf/huraim
RUN pip install -r /tf/huraim/requirements.txt

# This is copied from the base image reference here to allow a password
# https://hub.docker.com/layers/tensorflow/tensorflow/1.14.0-py3-jupyter/images/sha256-07321f6227cdb5c60e9276a26f5e8d3168237ad032203275ed27357289c4c6f4?context=explore
CMD "bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=huraim1337"