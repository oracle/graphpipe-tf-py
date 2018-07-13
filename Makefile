
ifeq ($(OS),Windows_NT)
  $(error Windows builds are not yet supported :()
else
	detected_OS := $(shell uname -s)
endif

ifeq ($(detected_OS),Linux)
BUILD_SCRIPT := build_linux.sh
test:
	docker run -it --rm \
		-v $(PWD):/src \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		themattrix/tox-base \
		/app/build_linux.sh
		#/bin/sh

build:
	docker run -it --rm \
		-v $(PWD):/app \
		-w /app \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		python:3.5 \
		/app/build_linux.sh $$(id -u):$$(id -g)
		#/bin/sh
endif

ifeq ($(detected_OS),Darwin)  # Mac OS X
BUILD_SCRIPT := build_darwin.sh
build:
	./build_darwin.sh
endif

