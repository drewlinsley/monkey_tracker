#!/bin/bash

echo "Which node do you want to use? [1, 2, 3]"
read node

for gpu in  0 1 2 3 4 5 6 7
do
	echo "GPU $gpu"
	# Build the image and label it as clickme-example on the Docker registry on p3
	nvidia-docker build -t serrep$node.services.brown.edu:5000/monkey_tracker .

	#Run the container
	nvidia-docker run -d --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs/cluster_projects/monkey_tracker serrep$node.services.brown.edu:5000/monkey_tracker sh run_hp_optim_worker.sh $gpu
done
