docker pull kuzminalexcmc/dockerhub:docker_flask

docker run -it -p 5001:5000 --rm -v "$PWD":/usr/src/app  kuzminalexcmc/dockerhub:docker_flask