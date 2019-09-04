from gym_brt.quanser.qube_interfaces import QubeSimulator

try:
    from gym_brt.quanser.qube_interfaces import QubeHardware
except ImportError:
    print("Warning: Can not import QubeHardware in quanser/__init__.py")
