build:
	sudo docker build -t rl_image:1.0 --cache-from rl_image:1.0 .

run:
	sudo docker run -u $(id -u):$(id -g) -v $(shell pwd):/app rl_image:1.0
