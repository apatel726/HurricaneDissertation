# Docker

This folder allows the development enviornment to be replicated exactly to the
extent of Docker. We can avoid development enviornment issues by deploying the 
Dockerfile included.

## Quickstart

Install Docker first. This Dockerfile should work on any OS. Please note that
the following commands should be run inside the `docker` folder.

```bash
docker build -t huraim .
docker run -it -p 8888:8888 huraim
```

After running the above commands, open up a web browser and go to
`localhost:8888`. It will ask for a token or password which is **huraim1337**.

### Notebook for Running the Universal Model

After sucessfully logging into the Jupyter notebook, we can run code inside the
Docker container. There is a notebook inside the container already to run the
universal model of HURAIM. The directory is `/tf/huraim/huraim.ipynb` where we
can run all cells and then change the input as necessary.

## Background

We are using the specific version of Tensorflow at 1.14 as defined in the 
`Dockerfile`. This creates a virtual machine based on Debian Linux with all the
dependencies pre-installed. This is from another Dockerfile created by the
Tensforflow team. We then build upon the Dockerfile with our own Dockerfile by
installing other Python libraries defined in the `requirements.txt`. From there,
we can run code but it's recommended to use Jupyter notebooks because it's more
developer friendly.