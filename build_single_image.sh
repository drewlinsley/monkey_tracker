#!/bin/bash

read -p "Enter the ID of the gpu you want to use: "  gpu
echo "Developing worker for gpu $gpu on p3."

# Build the image and label it as clickme-example on the Docker registry on p3
nvidia-docker build -t serrep3.services.brown.edu:5000/monkey_tracker .

#Run the container
nvidia-docker run -d --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs/cluster_projects/monkey_tracker serrep3.services.brown.edu:5000/monkey_tracker sh run_hp_optim_worker.sh $gpu
