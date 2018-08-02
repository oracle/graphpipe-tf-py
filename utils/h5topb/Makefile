IMAGE=graphpipe-h5topb

all:
	cp ../../examples/convert.py .
	docker build -t ${IMAGE} \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.
