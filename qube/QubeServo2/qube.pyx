from __future__ import print_function

cimport quanser_types as qt
cimport numpy as np
cimport hil

from helpers.error_codes import error_codes
import numpy as np
import time


ALLOWED_ANALOG_R_CHANNELS  = [0]
ALLOWED_ANALOG_W_CHANNELS  = [0]
ALLOWED_DIGITAL_W_CHANNELS = [0]
ALLOWED_ENCODER_R_CHANNELS = [0, 1]
ALLOWED_OTHER_R_CHANNELS   = [14000]
ALLOWED_LED_W_CHANNELS     = [11000, 11001, 11002]


def close_all_boards():
    hil.hil_close_all()


cdef class QubeServo2:
    cdef hil.t_card  board
    cdef hil.t_error result
    cdef hil.t_task  task

    cdef bint task_started

    cdef qt.t_uint32[::] analog_r_channels, analog_w_channels, \
            digital_w_channels, encoder_r_channels, other_r_channels, led_w_channels

    cdef qt.t_uint32 num_analog_r_channels, num_analog_w_channels, \
            num_digital_w_channels, num_encoder_r_channels, num_other_r_channels, \
            num_led_w_channels

    cdef qt.t_double[::] currents_r # num_analog_r_channels
    cdef qt.t_double[::] voltages_w # num_analog_w_channels
    cdef qt.t_boolean[::] enables_r # num_digital_w_channels
    cdef qt.t_int32[::] encoder_r_buffer # num_encoder_r_channels
    cdef qt.t_double[::] other_r_buffer # num_other_r_channels
    cdef qt.t_double[::] led_r_buffer # num_other_r_channels

    cdef qt.t_double frequency, period

    def __cinit__(self):
        board_type="qube_servo2_usb"
        board_identifier="0"
        try:
            result = hil.hil_open(board_type, board_identifier, &self.board)
            print("Successfully opened board")

        except:
            print("Failed to open board")

    def __dealloc__(self):
        """Make sure to free the board!"""
        hil.hil_close(self.board)

    def __init__(self, frequency=25):
        """
        Args:
        - analog_r_channels:  [INPUT]  a list of analog channels to use for commumication
        - analog_w_channels:  [OUTPUT] a list of analog channels to use for commumication
        - digital_w_channels: [INPUT]  a list of digital channels to use for commumication
        - encoder_r_channels: [INPUT]  a list of encoder channels to use for commumication
        - other_r_channels:   [INPUT]  a list of other channels to use for commumication
        - led_w_channels:     [OUTPUT] a list of led channels to use for commumication
        - board_type:       the name of the board (to find the board_type goto
                            file:///opt/quanser/hil_sdk/help/quarc_supported_quanser_cards.html
                            and find you card)
        - frequency:  Frequency of the reading/writing task (in Hz)
        """
        analog_r_channels  = [0]
        analog_w_channels  = [0]
        digital_w_channels = [0]
        encoder_r_channels = [0, 1]
        other_r_channels   = [14000]
        led_w_channels     = [11000, 11001, 11002]

        # Make sure channel names are valid
        self._validate("analog_r_channels", analog_r_channels)
        self._validate("analog_w_channels", analog_w_channels)
        self._validate("digital_w_channels", digital_w_channels)
        self._validate("encoder_r_channels", encoder_r_channels)
        self._validate("other_r_channels", other_r_channels)
        self._validate("led_w_channels", led_w_channels)

        # Convert the channels into numpy arrays which are then stored in memoryviews (to pass C buffers to the HIL API)
        self.num_analog_r_channels  = len(analog_r_channels)
        self.num_analog_w_channels  = len(analog_w_channels)
        self.num_digital_w_channels = len(digital_w_channels)
        self.num_encoder_r_channels = len(encoder_r_channels)
        self.num_other_r_channels   = len(other_r_channels)
        self.num_led_w_channels     = len(led_w_channels)
        self.analog_r_channels  = np.array(analog_r_channels, dtype=np.uint32)
        self.analog_w_channels  = np.array(analog_w_channels, dtype=np.uint32)
        self.digital_w_channels = np.array(digital_w_channels, dtype=np.uint32)
        self.encoder_r_channels = np.array(encoder_r_channels, dtype=np.uint32)
        self.other_r_channels   = np.array(other_r_channels, dtype=np.uint32)
        self.led_w_channels     = np.array(led_w_channels, dtype=np.uint32)

        self.frequency = frequency
        self.task_started = False

    def __enter__(self):
        """Start the hardware in a deterministic way (all motors, encoders, etc at 0)"""
        # Create a memoryview for currents
        self.currents_r = np.zeros(self.num_analog_r_channels,
                                    dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for -ometers
        self.other_r_buffer = np.zeros(self.num_other_r_channels,
                                    dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for leds
        self.led_r_buffer = np.zeros(self.num_led_w_channels,
                                    dtype=np.float64) # t_double is 64 bits

        # Set all motor voltages_w to 0
        self.voltages_w = np.zeros(self.num_analog_w_channels,
                                    dtype=np.float64) # t_double is 64 bits
        result = hil.hil_write_analog(self.board,
                                    &self.analog_w_channels[0],
                                    self.num_analog_w_channels,
                                    &self.voltages_w[0])
        if result < 0:
            self._print_possible_error(result)
            raise IOError("Could not set voltages_w to 0")

        # Set the encoder encoder_r_buffer to 0
        self.encoder_r_buffer = np.zeros(self.num_encoder_r_channels,
                                    dtype=np.int32) # t_int32 is 32 bits
        result = hil.hil_set_encoder_counts(self.board,
                                    &self.encoder_r_channels[0],
                                    self.num_encoder_r_channels,
                                    &self.encoder_r_buffer[0])
        if result < 0:
            self._print_possible_error(result)
            raise IOError("Could not set encoder buffer to 0")

        # Enables_r all the motors
        self.enables_r = np.ones(self.num_digital_w_channels,
                            dtype=np.int8) # t_bool is char (8 bits)
        result = hil.hil_write_digital(self.board,
                        &self.digital_w_channels[0],
                        self.num_digital_w_channels,
                        &self.enables_r[0])
        if result < 0:
            self._print_possible_error(result)
            raise IOError("Could not set enables_r to 0")

        return self

    def __exit__(self, type, value, traceback):
        """Make sure hardware turns off safely"""
        self._stop_task()

        # Set the motor voltages_w to 0
        self.voltages_w = np.zeros(self.num_analog_w_channels,
                                    dtype=np.float64) # t_double is 64 bits
        hil.hil_write_analog(self.board,
                                &self.analog_w_channels[0],
                                self.num_analog_w_channels,
                                &self.voltages_w[0])

        # Disable all the motors
        self.enables_r = np.zeros(self.num_digital_w_channels,
                                dtype=np.int8) # t_bool is char (8 bits)
        hil.hil_write_digital(self.board,
                                &self.digital_w_channels[0],
                                self.num_digital_w_channels,
                                &self.enables_r[0])

    @staticmethod
    def _print_possible_error(int result):
        """If there is an error, print the error code. TODO: get error codes from HIL API"""
        if result < 0:
            print(error_codes[-result])

    @staticmethod
    def _validate(channel_type, channels):
        """Make sure the channels given are valid for the hardware"""
        def check_channels(allowed_channels, channels):
            if len(channels) > len(allowed_channels):
                raise ValueError("Too many channels given! Channels: {}".format(channels))
            elif len(set(channels)) != len(channels):
                raise ValueError("Channels repeated! Channels: {}".format(channels))
            elif len(channels) == len(allowed_channels):
                if set(channels) != set(allowed_channels):
                    return
            else:
                for el in channels:
                    if el not in set(allowed_channels):
                        raise ValueError("Channels that are not in allowed channels! Channels: {}, Allowed Channels: {}".format(channels, allowed_channels))
                return
        if channel_type == "analog_r_channels":
            allowed_channels = ALLOWED_ANALOG_R_CHANNELS
        elif channel_type == "analog_w_channels":
            allowed_channels = ALLOWED_ANALOG_W_CHANNELS
        elif channel_type == "digital_w_channels":
            allowed_channels = ALLOWED_DIGITAL_W_CHANNELS
        elif channel_type == "encoder_r_channels":
            allowed_channels = ALLOWED_ENCODER_R_CHANNELS
        elif channel_type == "other_r_channels":
            allowed_channels = ALLOWED_OTHER_R_CHANNELS
        elif channel_type == "led_w_channels":
            allowed_channels = ALLOWED_LED_W_CHANNELS
        else:
            raise ValueError("Channel type '{}' is invalid".format(channel_type))

    def _create_task(self):
        """Start a task reads and writes at fixed intervals"""

        result =  hil.hil_task_create_reader(self.board,
             10000, # The size of the internal buffer (making this >> 1 prevents
                    # error 111 but may also occasionally miss a read of state)
             &self.analog_r_channels[0],
             self.num_analog_r_channels, # Analog inp channels (motor current sense)
             &self.encoder_r_channels[0],
             self.num_encoder_r_channels, # Encoder inp channels (encoder counts, pitch, yaw)
             NULL, 0, # Digital input channels (errors and faults)
             &self.other_r_channels[0],
             self.num_other_r_channels, # Other inp channels (gyro, accelerometer, & tach)
             &self.task)

        self._print_possible_error(result)

        # Start the task
        result = hil.hil_task_start(self.task, hil.HARDWARE_CLOCK_0, self.frequency, -1)
        self._print_possible_error(result)

    def _stop_task(self):
        if self.task_started:
            hil.hil_task_flush(self.task)
            hil.hil_task_stop(self.task)
            hil.hil_task_delete(self.task)

    def action(self, voltages_w):
        """Make sure you get safe data!"""
        # If it's the first time running action, then start the background r/w task
        if not self.task_started:
            self._create_task()
            self.task_started = True

        if isinstance(voltages_w, list):
            voltages_w = np.array(voltages_w, dtype=np.float64)
        assert isinstance(voltages_w, np.ndarray)
        assert voltages_w.shape == (self.num_analog_w_channels,)
        assert voltages_w.dtype == np.float64
        for i in range(self.num_analog_w_channels):
            assert -15.0 <= voltages_w[i] <= 15.0 # Operating range

        return self._action(voltages_w)

    def _action(self, np.ndarray[qt.t_double, ndim=1, mode="c"] voltages_w not None):
        """Perform actions on the device (voltages_w must always be ndarray!)"""
        # First read using task_read (blocking call that enforces timing)
        samples = hil.hil_task_read(self.task, 1,
            &self.currents_r[0],       # Analog input channels (motor current sense)
            &self.encoder_r_buffer[0], # Encoder input channels (encoder counts, pitch, yaw)
            NULL,                      # Digital input channels (errors and faults)
            &self.other_r_buffer[0])   # Other input channels (gyro, accelerometer, & tach)
        self._print_possible_error(samples)

        # Then write voltages_w calculated for previous time step
        self.voltages_w = voltages_w
        result = hil.hil_write_analog(self.board, &self.analog_w_channels[0],
                self.num_analog_w_channels, &self.voltages_w[0])
        self._print_possible_error(result)

        return np.asarray(self.currents_r), \
                np.asarray(self.encoder_r_buffer), \
                np.asarray(self.other_r_buffer)
