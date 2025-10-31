# avcServoCode: Visual Servoing Demo for Jetson Orin

This is a simple visual servoing demo. A **Python** script on the Jetson Orin (using OpenCV/Jetson Inference) captures a camera feed, calculates a positional error, and sends it to an **Arduino UNO**. A **C++** program on the UNO then runs a control loop, driving a servo to automatically pan the camera and keep the target centered.

## 🚀 Requirements

* **Hardware:** NVIDIA Jetson Orin, Arduino UNO, Servo Motor, Camera (CSI/USB)
* **Software:** NVIDIA JetPack 5.x, OpenCV, Jetson Inference, and a trained dataset.
