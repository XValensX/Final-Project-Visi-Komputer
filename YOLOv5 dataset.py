from roboflow import Roboflow

rf = Roboflow(api_key="AEbtnlIwYxDrUoCYM6Pm")
project = rf.workspace("bike-helmets").project("bike-helmet-detection-2vdjo")
version = project.version(2)
dataset = version.download("yolov5")
