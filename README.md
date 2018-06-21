# Quanser OpenAI Driver
Has an OpenAI Gym wrapper for the Quanser Qube Servo 2 and Quanser Aero


# Installation
This was _ONLY_ tested on Ubuntu 16.04 LTS and Ubunbtu 18.04 LTS using Python 2.7. <br>
To compile the Cython wrapper, you need a to have the Cython package installed (we recommend using virtualenv). <br>
Cython is not needed once you have the wrapper compiled (only Python). <br>

To compile and run a test
1. Install the HIL SDK
	- `cd <installation-directory>/quanser-openai-driver/hil_sdk_linux_x86_64 && ./setup_hil_sdk`
2. Compile the Cython wrapper
	- `cd <installation-directory>/quanser-openai-driver/gym-enviroments/gym_brt/envs/QuanserWrapper && make`
3. Install gym_qube
	- `cd <installation-directory>/quanser-openai-driver/gym-enviroments && pip install -e .`
4. Run the random agent test
	- For the Qube: `python <installation-directory>/quanser-openai-driver/tests/test_inverted_pendulum.py`
	- For the Aero: `python <installation-directory>/quanser-openai-driver/tests/test_aero.py`

The `hil_sdk_linux_x86_64` directory contains the installation files for the HIL SDK (Copyright (C) Quanser Inc.) <br>
The `gym-enviroments` directory contains a Python and OpenAI Gym wrapper around the HIL SDK

![Qube Standing Up](/QUBE-Servo_2_angled_pendulum.jpg?raw=true)



