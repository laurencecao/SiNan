#!/bin/bash


docker run --runtime nvidia --rm -it -v .:/code -e JUPYTER_PORT=8000 -e JUPYTER_PASSWORD="mypassword" -e USER_PASSWORD="unsloth2024" -p 28000:8000 docker.1ms.run/unsloth/unsloth:2026.2.1-pt2.9.0-cu12.8-moe-optimized-training bash
