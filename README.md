MNIST Project

A simple project for classifying handwritten digits from the MNIST dataset (digits 0–9, 28×28 grayscale images).

About

MNIST contains 70,000 small grayscale images of handwritten digits, each labeled with the correct digit. The dataset is usually split into 60,000 training images and 10,000 test images.

The project trains a model to look at an image and predict which digit it shows. The model learns from the labeled training images, and its accuracy is then measured on the test images it has never seen.

MNIST is widely used because it is:

Small and fast, so models train on an ordinary laptop in minutes.
Clean, with centered, correctly labeled digits, so beginners can focus on modeling.
A common benchmark, so results can be compared across projects.
A good introduction to the standard workflow of loading data, building a model, training, and evaluating.

A simple model typically reaches about 97–99% test accuracy.

Setup
Clone or download the project and go to its folder:
bash
cd mnist-project
(Optional) Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install the dependencies:
bash
pip install -r requirements.txt
