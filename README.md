# avcServoCode: Visual Servoing Demo for Jetson Orin 

This project is a simple visual servoing demonstration for the NVIDIA Jetson Orin. It uses a camera to detect a target and a servo motor to automatically pan the camera to keep the target centered in the frame.

## How It Works

The system is split into two primary components that run concurrently:

1.  **Vision (Python):** The Python script (`.py`) captures the camera feed, calculates a positional error (e.g., the difference between a target's coordinates and the center of the frame), and sends this error value.

2.  **Control (C++):** The C++ program (`.cpp`) receives the error value and implements a control loop (e.g., a simple P-controller) to generate a PWM signal. This signal drives the servo, turning the camera to correct the error.

## Requirements

* **Hardware:**
    * NVIDIA Jetson Orin
    * UNO
    * Servo Motor
    * Camera (e.g., CSI or USB)
* **Software:**
    * NVIDIA JetPack 5.x
    * Open CV, Jetson Inference lib (DustyNV), and have a data set.
