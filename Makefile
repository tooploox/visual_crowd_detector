IMAGE_NAME=crowd_detector
PORT ?= 9999

build:
	docker build -t $(IMAGE_NAME) .

dev:
	docker run --rm -ti  \
		--runtime=nvidia \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME)

dev_gui:
	docker run --rm -ti \
		-v $$PWD/:/project \
		--runtime="nvidia" \
		--env="DISPLAY" \
		--env="QT_X11_NO_MITSHM=1" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-w="/project" \
		$(IMAGE_NAME)

lab:
	docker run --rm -ti  \
		--runtime=nvidia \
		-p $(PORT):$(PORT) \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME) \
		jupyter lab --ip=0.0.0.0 --port=$(PORT) --allow-root --no-browser