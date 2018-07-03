test:
	docker run -it --rm \
		-v $(PWD):/src \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		themattrix/tox-base \
		/app/manually.sh
		#/bin/sh

build:
	docker run -it --rm \
		-v $(PWD):/app \
		-w /app \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		python:3.5 \
		/app/manually.sh $$(id -u):$$(id -g)
		#/bin/sh
