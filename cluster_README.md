# Running

Use `build_image.sh` to generate the Docker image, and `kubectl apply -f
kube/job.yaml` to schedule it on Kubernetes.

See the [Kubernetes Job
documentation](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/)
for more advanced scheduling (ex. distributing work through Kubernetes).

MANUAL STEPS

# Build the image and label it as clickme-example on the Docker registry on p3
nvidia-docker build -t serrep3.services.brown.edu:5000/monkey_tracker .

#Run the container
nvidia-docker run -it --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs serrep3.services.brown.edu:5000/monkey_tracker bash
nvidia-docker run -it --volume /media/data_cifs:/media/data_cifs --workdir /media/data_cifs serrep3.services.brown.edu:5000/monkey_tracker python train_and_eval_joints.py
