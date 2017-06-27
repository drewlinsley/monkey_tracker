#!/bin/bash

set -eou pipefail
trap 'kill $(jobs -p)' EXIT

# Connect to Docker on p3 to get around the firewall
ssh -N -L localhost:2375:/var/run/docker.sock serrep3.services.brown.edu &
sleep 1
export DOCKER_HOST=tcp://localhost:2375

# Build the image and label it as clickme-example on the Docker registry on p3
docker build -t serrep3.services.brown.edu:5000/clickme-example .

# And push to the registry
docker push serrep3.services.brown.edu:5000/clickme-example
