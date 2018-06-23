from __future__ import print_function

cimport quanser_types as qt
cimport numpy as np
cimport hil

from gym_brt.envs.QuanserWrapper.helpers.error_codes import print_possible_error

import numpy as np
import time


cdef class QuanserWrapper:
    cdef hil.t_card  board
    cdef hil.t_error result
    cdef hil.t_task  task

    cdef bint task_started

    cdef qt.t_uint32[::] analog_r_channels
    cdef qt.t_uint32 num_analog_r_channels
    cdef qt.t_double[::] currents_r

    cdef qt.t_uint32[::] analog_w_channels
    cdef qt.t_uint32 num_analog_w_channels
    cdef qt.t_double[::] voltages_w
    
    cdef qt.t_uint32[::] digital_w_channels
    cdef qt.t_uint32 num_digital_w_channels
    cdef qt.t_boolean[::] enables_r
    
    cdef qt.t_uint32[::] encoder_r_channels
    cdef qt.t_uint32 num_encoder_r_channels
    cdef qt.t_int32[::] encoder_r_buffer
    
    cdef qt.t_uint32[::] other_r_channels
    cdef qt.t_uint32 num_other_r_channels
    cdef qt.t_double[::] other_r_buffer
    
    cdef qt.t_uint32[::] led_w_channels
    cdef qt.t_uint32 num_led_w_channels
    cdef qt.t_double[::] led_w_buffer

    cdef qt.t_double frequency, period, prev_action_time

    def __init__(self,
                 analog_r_channels,
                 analog_w_channels,
                 digital_w_channels,
                 encoder_r_channels,
                 other_r_channels,
                 led_w_channels,
                 frequency=1000):
        """
        Args:
            - analog_r_channels : [INPUT]  a list of analog channels to use for commumication
            - analog_w_channels : [OUTPUT] a list of analog channels to use for commumication
            - digital_w_channels: [INPUT]  a list of digital channels to use for commumication
            - encoder_r_channels: [INPUT]  a list of encoder channels to use for commumication
            - other_r_channels  : [INPUT]  a list of other channels to use for commumication
            - led_w_channels    : [OUTPUT] a list of led channels to use for commumication
            - board_type        : the name of the board (to find the board_type goto 
                                  file:///opt/quanser/hil_sdk/help/quarc_supported_quanser_cards.html
                                  and find you card)
            - Frequency         : Frequency of the reading/writing task (in Hz)
        """
        # Convert the channels into numpy arrays which are then stored in 
        # memoryviews (to pass C buffers to the HIL API)
        self.num_analog_r_channels = len(analog_r_channels)
        self.num_analog_w_channels = len(analog_w_channels)
        self.num_digital_w_channels = len(digital_w_channels)
        self.num_encoder_r_channels = len(encoder_r_channels)
        self.num_other_r_channels = len(other_r_channels)
        self.num_led_w_channels = len(led_w_channels)
        self.analog_r_channels = np.array(analog_r_channels, dtype=np.uint32)
        self.analog_w_channels = np.array(analog_w_channels, dtype=np.uint32)
        self.digital_w_channels = np.array(digital_w_channels, dtype=np.uint32)
        self.encoder_r_channels = np.array(encoder_r_channels, dtype=np.uint32)
        self.other_r_channels = np.array(other_r_channels, dtype=np.uint32)
        self.led_w_channels = np.array(led_w_channels, dtype=np.uint32)

        self.frequency = frequency
        self.task_started = False

    def __enter__(self):
        """
        Start the hardware in a deterministic way (all motors, encoders, etc
        at 0)
        """
        # Create a memoryview for currents
        self.currents_r = np.zeros(
            self.num_analog_r_channels,
            dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for -ometers
        self.other_r_buffer = np.zeros(
            self.num_other_r_channels,
            dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for leds
        self.led_w_buffer = np.zeros(
            self.num_led_w_channels,
            dtype=np.float64) # t_double is 64 bits

        # Set all motor voltages_w to 0
        self.voltages_w = np.zeros(
            self.num_analog_w_channels,
            dtype=np.float64) # t_double is 64 bits
        result = hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])
        print_possible_error(result)

        # Set the encoder encoder_r_buffer to 0
        self.encoder_r_buffer = np.zeros(
            self.num_encoder_r_channels,
            dtype=np.int32) # t_int32 is 32 bits
        result = hil.hil_set_encoder_counts(
            self.board,
            &self.encoder_r_channels[0],
            self.num_encoder_r_channels,
            &self.encoder_r_buffer[0])
        print_possible_error(result)

        # Enables_r all the motors
        self.enables_r = np.ones(
            self.num_digital_w_channels,
            dtype=np.int8) # t_bool is char (8 bits)
        result = hil.hil_write_digital(
            self.board,
            &self.digital_w_channels[0],
            self.num_digital_w_channels,
            &self.enables_r[0])
        print_possible_error(result)

        return self

    def __exit__(self, type, value, traceback):
        """Make sure hardware turns off safely"""
        self._stop_task()

        # Set the motor voltages_w to 0
        self.voltages_w = np.zeros(
            self.num_analog_w_channels,
            dtype=np.float64) # t_double is 64 bits
        hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])

        # Disable all the motors
        self.enables_r = np.zeros(
            self.num_digital_w_channels,
            dtype=np.int8) # t_bool is char (8 bits)
        hil.hil_write_digital(
            self.board,
            &self.digital_w_channels[0],
            self.num_digital_w_channels,
            &self.enables_r[0])

    def _create_task(self):
        """Start a task reads and writes at fixed intervals"""

        result =  hil.hil_task_create_reader(
            self.board,
            1000, # The size of the internal buffer (making this >> 1 
                  # prevents error 111 but may also occasionally miss a read
                  # of state)
            &self.analog_r_channels[0], self.num_analog_r_channels,
            &self.encoder_r_channels[0], self.num_encoder_r_channels,
            NULL, 0,
            &self.other_r_channels[0], self.num_other_r_channels,
            &self.task)
        print_possible_error(result)

        # Start the task
        result = hil.hil_task_start(
            self.task,
            hil.HARDWARE_CLOCK_0,
            self.frequency,
            -1) # Read continuously 
        print_possible_error(result)

    def _stop_task(self):
        if self.task_started:
            hil.hil_task_flush(self.task)
            hil.hil_task_stop(self.task)
            hil.hil_task_delete(self.task)

    def action(self, voltages_w):
        """Make sure you get safe data!"""
        
        # If it's the first time running action, then start the background r/w 
        # task
        if not self.task_started:
            self._create_task()
            self.task_started = True

        if isinstance(voltages_w, list):
            voltages_w = np.array(voltages_w, dtype=np.float64)
        assert isinstance(voltages_w, np.ndarray)
        assert voltages_w.shape == (self.num_analog_w_channels,)
        assert voltages_w.dtype == np.float64
        for i in range(self.num_analog_w_channels):
            assert -25.0 <= voltages_w[i] <= 25.0 # Operating range

        return self._action(voltages_w)

    def _action(self,
                np.ndarray[qt.t_double, ndim=1, mode="c"] voltages_w not None):
        """Perform actions on the device (voltages_w must always be ndarray!)"""
        # First read using task_read (blocking call that enforces timing)
        samples = hil.hil_task_read(
            self.task,
            1, # Number of samples to read
            &self.currents_r[0],
            &self.encoder_r_buffer[0],
            NULL,
            &self.other_r_buffer[0])
        print_possible_error(samples)

        # Then write voltages_w calculated for previous time step
        self.voltages_w = voltages_w
        result = hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])
        print_possible_error(result)

        return np.asarray(self.currents_r), np.asarray(self.encoder_r_buffer), \
            np.asarray(self.other_r_buffer)


cdef class QuanserAero(QuanserWrapper):
    def __cinit__(self):
        board_type = "quanser_aero_usb"
        board_identifier = "0"
        result = hil.hil_open(board_type, board_identifier, &self.board)
        print_possible_error(result)
        if result < 0:
            raise IOError

    def __init__(self, frequency=100):
        analog_r_channels = [0, 1]
        analog_w_channels = [0, 1]
        digital_w_channels = [0, 1]
        encoder_r_channels = [0, 1, 2, 3]
        other_r_channels  = [3000, 3001, 3002, 4000, 4001, 4002, 14000, 14001, \
                             14002, 14003]
        led_w_channels    = [11000, 11001, 11002]

        super(QuanserAero, self).__init__(
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)

    def __dealloc__(self):
        """Make sure to free the board!"""
        print("In QuanserAero __dealloc__")
        hil.hil_close(self.board)


cdef class QubeServo2(QuanserWrapper):
    def __cinit__(self):
        board_type="qube_servo2_usb"
        board_identifier="0"
        result = hil.hil_open(board_type, board_identifier, &self.board)
        print_possible_error(result)

    def __init__(self, frequency=100):
        analog_r_channels = [0]
        analog_w_channels = [0]
        digital_w_channels = [0]
        encoder_r_channels = [0, 1]
        other_r_channels  = [14000]
        led_w_channels    = [11000, 11001, 11002]

        super(QubeServo2, self).__init__(
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)

    def __dealloc__(self):
        """Make sure to free the board!"""
        hil.hil_close(self.board)



