FROM serrep3.services.brown.edu:5000/tensorflow

MAINTAINER Ben Navetta <benjamin_navetta@brown.edu>

RUN pip install scipy

COPY . .

CMD ["python", "train_and_eval_joints.py"]
