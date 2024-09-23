# cam-callib-openpilot

### Context
The devices that run openpilot are not mounted perfectly. The camera is not exactly aligned to the vehicle. There is some pitch and yaw angle between the camera of the device and the vehicle, which can vary between installations. Estimating these angles is essential for accurate control of the vehicle. The best way to start estimating these values is to predict the direction of motion in camera frame. More info can be found in this readme.
