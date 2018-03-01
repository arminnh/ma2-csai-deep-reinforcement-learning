#!/usr/bin/env bash
sudo cp resolv.conf /etc/resolv.conf
docker run -ti -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.flashgames:0.20.28
