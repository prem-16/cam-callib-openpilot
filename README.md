# cam-callib-openpilot




# Pitch and Yaw Estimation from Video Frames

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The devices that run openpilot are not mounted perfectly. The camera is not exactly aligned to the vehicle. There is some pitch and yaw angle between the camera of the device and the vehicle, which can vary between installations. Estimating these angles is essential for accurate control of the vehicle. The best way to start estimating these values is to predict the direction of motion in camera frame. More info can be found in this readme.

This project estimates the pitch and yaw angles of a camera from video frames using computer vision techniques, specifically line detection and optical flow. It aims to provide a robust method for angle estimation in dynamic environments.

## Features

- Line detection using Hough Transform
- Optical flow calculation 
- Vanishing point estimation based on detected lines

## Installation

To get started with this project, clone the repository and install the required packages.


