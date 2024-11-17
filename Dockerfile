FROM python:3.10

RUN apt-get update\
  && apt-get install \
        gcc 

RUN pip3 install stim

RUN pip3 install numpy scipy pymatching 

RUN pip3 install git+https://github.com/JiakaiW/FlexibleQECSim

# Use: in terminal run
# docker build --no-cache -t surfacesimulationtest:v13 .