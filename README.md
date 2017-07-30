# Automatic monkey pose discovery

## Copy the 'config.py.template' file as 'config.py' and adjust it to match your system.
## Do the same for 'kinect_config.py.template', copy it as 'kinect_config.py', and adjust it to match your system.

---
# Methods overview

The aim of this project is to automatically estimate monkey pose by localizing each of its joints. In the current implementation, this main learning problem involves estimating x/y coordinates of 23 joints on a monkey's body. Each of the steps that we use to train a system on this problem is outlined below. Click each bullet to find more details on each section.

1. Generating big data with computer rendering software for training a CNN to estimate monkey pose.

2. Converting computer rendered images and annotations into "tfrecords", which support CNN training.

3. Processing Kinect data for human-in-the-loop annotations on vid-int.

4. Training a CNN to estimate pose on a combination of computer renders and Kinect data.

5. Applying the models to held-out datasets.

---
1. Generating big data with computer rendering software for training a CNN to estimate monkey pose.


---
1. Generating big data with computer rendering software for training a CNN to estimate monkey pose.


---
# Model training on single machine


---
# Model training on cluster

---
# Testing 

---
#TODO


# This pro


# Training the model

1. Render tons of exemplars with Maya -- @dmurphy can you fill this section out??



# Testing with Kinect data
1. Preprocess data with `python preprocess_kinect_with_model.py`



