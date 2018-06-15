# quanser-openai-driver
Has an openai gym wrapper for the quanser qube and quanser aero


# Installation
1. Install the HIL SDK
	- `cd <installation-directory>/quanser-openai-driver/hil_sdk_linux_x86_64 && ./setup_hil_sdk`
2. Install gym_qube
	- `cd <installation-directory>/quanser-openai-driver/gym-enviroments && pip install -e .`
3. Run the random agent test
	- `python <installation-directory>/quanser-openai-driver/tests/test_qube_env.py`


The `hil_sdk_linux_x86_64` directory contains the installation files for the HIL SDK (Copyright (C) Quanser Inc.)
The `qube` directory contains a Python and OpenAI Gym wrapper around the HIL SDK

![Alt text](/QUBE-Servo_2_angled_pendulum.jpg?raw=true "Qube Standing Up")



