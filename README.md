# Docker + Keras tuner
This is a simple demo for using keras tuner in a Docker container.

To build the image:

`$ docker build -t my_kerastuner .`

To run the image:

`$ docker run --mount type=bind,src="$(pwd)",target=/workdir my_kerastuner`

To prevent permission related issues:
```
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker
```