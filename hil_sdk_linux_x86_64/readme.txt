1) Quanser HIL SDK Installation 

- To install the Quanser HIL SDK on Linux:

chmod a+x setup_hil_sdk uninstall_hil_sdk
sudo ./setup_hil_sdk

Type yes in response to the license agreement prompt if you agree with the license.

- To uninstall the HIL SDK software, run the uninstall_hil_sdk command (in 
/usr/sbin).

- Our testing was done with the out-of-the-box 64-bit Ubuntu 16.04.4.



2) Quanser HIL SDK Documentation  

- When the installation is complete, the HIL SDK documentation is available by running the HIL SDK application using the Ubuntu Launcher (or by opening /opt/quanser/hil_sdk/help/index.html in FireFox). 


- In particular, the following documentation page explains how to build a Linux application using the HIL SDK and where to find the examples:

HIL SDK/Hardware/HIL API/Getting Started with the HIL C API/Setting up the C/C++ application to use the HIL C API/Setting up to use the HIL C API in Linux


3) Quanser HIL SDK Examples 

- HIL SDK examples may be found under /opt/quanser/hil_sdk/examples.

- In particular, the 3 following examples already interface to the Quanser QUBE-SERVO2-USB:  
/opt/quanser/hil_sdk/examples/C/hardware/qube_servo2_usb_read_example
/opt/quanser/hil_sdk/examples/C/hardware/qube_servo2_usb_write_example
/opt/quanser/hil_sdk/examples/C/hardware/qube_servo2_usb_control_example

