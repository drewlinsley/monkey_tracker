FROM serrep3.services.brown.edu:5000/tensorflow

MAINTAINER Ben Navetta <benjamin_navetta@brown.edu>

RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

COPY . .

CMD ["python", "train_and_eval_joints.py"]
