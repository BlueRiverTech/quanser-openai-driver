from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

cimport gym_brt.quanser.quanser_wrapper.quanser_types as qt
cimport gym_brt.quanser.quanser_wrapper.hil as hil
cimport numpy as np

from gym_brt.quanser.quanser_wrapper.error_codes import error_codes
import numpy as np


cdef print_possible_error(int result):
    '''If there is an error, print the error code'''
    if result < 0:
        print(error_codes[-result])


cdef class QuanserWrapper:
    cdef hil.t_card board
    cdef hil.t_task task

    cdef qt.t_uint32[::] analog_r_channels
    cdef qt.t_uint32 num_analog_r_channels
    cdef qt.t_double[::] currents_r

    cdef qt.t_uint32[::] analog_w_channels
    cdef qt.t_uint32 num_analog_w_channels
    cdef qt.t_double[::] voltages_w

    cdef qt.t_uint32[::] digital_w_channels
    cdef qt.t_uint32 num_digital_w_channels
    cdef qt.t_boolean[::] enables_w

    cdef qt.t_uint32[::] encoder_r_channels
    cdef qt.t_uint32 num_encoder_r_channels
    cdef qt.t_int32[::] encoder_r_buffer

    cdef qt.t_uint32[::] other_r_channels
    cdef qt.t_uint32 num_other_r_channels
    cdef qt.t_double[::] other_r_buffer

    cdef qt.t_uint32[::] led_w_channels
    cdef qt.t_uint32 num_led_w_channels
    cdef qt.t_double[::] led_w_buffer

    cdef qt.t_double frequency, safe_operating_voltage
    cdef qt.t_int samples_overflowed
    cdef bint task_started

    def __init__(self,
                 safe_operating_voltage,
                 analog_r_channels,
                 analog_w_channels,
                 digital_w_channels,
                 encoder_r_channels,
                 other_r_channels,
                 led_w_channels,
                 frequency=1000):
        self.safe_operating_voltage = safe_operating_voltage
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
        '''Start the hardware in a deterministic way (all motors,
        encoders, etc at 0)
        '''
        # Create memoryviews for read buffers
        self.currents_r = np.zeros(
            self.num_analog_r_channels,
            dtype=np.float64)  # t_double is 64 bits
        self.other_r_buffer = np.zeros(
            self.num_other_r_channels,
            dtype=np.float64)  # t_double is 64 bits

        # Set motor voltages_w and encoders to 0
        self.voltages_w = np.zeros(
            self.num_analog_w_channels,
            dtype=np.float64)  # t_double is 64 bits
        result = hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])
        print_possible_error(result)
        self.encoder_r_buffer = np.zeros(
            self.num_encoder_r_channels,
            dtype=np.int32)  # t_int32 is 32 bits
        result = hil.hil_set_encoder_counts(
            self.board,
            &self.encoder_r_channels[0],
            self.num_encoder_r_channels,
            &self.encoder_r_buffer[0])
        print_possible_error(result)

        # Set LED on to white (represents opening qube)
        self.led_w_buffer = np.ones(
            self.num_led_w_channels,
            dtype=np.float64)  # t_double is 64 bits
        result = hil.hil_write_other(
            self.board,
            &self.led_w_channels[0],
            self.num_led_w_channels,
            &self.led_w_buffer[0])
        print_possible_error(result)

        # Enables_r all the motors
        self.enables_w = np.ones(
            self.num_digital_w_channels,
            dtype=np.int8)  # t_bool is char 8 bits
        result = hil.hil_write_digital(
            self.board,
            &self.digital_w_channels[0],
            self.num_digital_w_channels,
            &self.enables_w[0])
        print_possible_error(result)

        return self

    def __exit__(self, type, value, traceback):
        '''Make sure hardware turns off safely'''
        self._stop_task()

        # Set the motor voltages_w to 0
        self.voltages_w = np.zeros(
            self.num_analog_w_channels, dtype=np.float64)  # t_double is 64 bits
        hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])

        # Set LED on to default color (red)
        self.led_w_buffer = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64)  # t_double is 64 bits
        result = hil.hil_write_other(
            self.board,
            &self.led_w_channels[0],
            self.num_led_w_channels,
            &self.led_w_buffer[0])
        print_possible_error(result)


        # Disable all the motors
        self.enables_w = np.zeros(
            self.num_digital_w_channels, dtype=np.int8)  # t_bool is char 8 bits
        hil.hil_write_digital(
            self.board,
            &self.digital_w_channels[0],
            self.num_digital_w_channels,
            &self.enables_w[0])

        hil.hil_close(self.board)  # Safely close the board

    def _create_task(self):
        '''Start a task reads and writes at fixed intervals'''
        result =  hil.hil_task_create_reader(
            self.board,
            1,  # Read 1 sample at a time
            &self.analog_r_channels[0], self.num_analog_r_channels,
            &self.encoder_r_channels[0], self.num_encoder_r_channels,
            NULL, 0,
            &self.other_r_channels[0], self.num_other_r_channels,
            &self.task)
        print_possible_error(result)

        # Allow discarding of old samples after missed reads
        hil.hil_task_set_buffer_overflow_mode(
            self.task, hil.BUFFER_MODE_OVERWRITE_ON_OVERFLOW)

        # Start the task
        result = hil.hil_task_start(
            self.task,
            hil.HARDWARE_CLOCK_0,
            self.frequency,
            -1)  # Read continuously
        print_possible_error(result)
        if result < 0:
            raise ValueError('Could not start hil task')

        self.task_started = True

    def _stop_task(self):
        if self.task_started:
            self.task_started = False
            hil.hil_task_flush(self.task)
            hil.hil_task_stop(self.task)
            hil.hil_task_delete(self.task)

    def reset_encoders(self, channels=None):
        '''Reset all or a few of the encoders'''
        if channels is None:
            # Set the entire encoder encoder_r_buffer to 0
            self.encoder_r_buffer = np.zeros(
                self.num_encoder_r_channels,
                dtype=np.int32)  # t_int32 is 32 bits
        else:
            # Set only specific encoders to 0, while leaving the others
            for channel in channels:
                # Check if the channel is valid (in the available encoder
                # channels for the hardware)
                if channel not in self.encoder_r_channels:
                    raise ValueError('Channel: {} is not a possible channel on '
                        + 'this hardware.')
                self.encoder_r_buffer[channel] = 0

        result = hil.hil_set_encoder_counts(
            self.board,
            &self.encoder_r_channels[0],
            self.num_encoder_r_channels,
            &self.encoder_r_buffer[0])
        print_possible_error(result)

    def action(self, voltages_w, led_w=None):
        # If it's the first time running action, then start the background r/w
        # task
        if not self.task_started:
            self._create_task()

        # Ensure safe voltage data
        if isinstance(voltages_w, list):
            voltages_w = np.array(voltages_w, dtype=np.float64)
        assert isinstance(voltages_w, np.ndarray)
        assert voltages_w.shape == (self.num_analog_w_channels,)
        assert voltages_w.dtype == np.float64
        for i in range(self.num_analog_w_channels):
            assert -self.safe_operating_voltage <= voltages_w[i] <= \
                    self.safe_operating_voltage

        if led_w is not None:
            # Ensure safe LED data
            if isinstance(led_w, list):
                led_w = np.array(led_w, dtype=np.float64)
            assert led_w.shape == (self.num_led_w_channels,)
            assert led_w.dtype == np.float64
            for i in range(self.num_led_w_channels):
                assert 0.0 <= led_w[i] <= 1.0  # HIL uses RGB scaled from 0-1
            self._set_led(led_w)  # An immediate write to LED (not timed task)

        return self._action(voltages_w)

    def _action(self,
                np.ndarray[qt.t_double, ndim=1, mode='c'] voltages_w not None):
        samples_read = hil.hil_task_read(
            self.task,
            1, # Number of samples to read
            &self.currents_r[0],
            &self.encoder_r_buffer[0],
            NULL,
            &self.other_r_buffer[0])
        if samples_read < 0:
            print_possible_error(samples_read)

        samples_overflowed = hil.hil_task_get_buffer_overflows(self.task)
        if samples_overflowed > self.samples_overflowed:
            print('Missed {} samples'.format(
                samples_overflowed - self.samples_overflowed))
            self.samples_overflowed = samples_overflowed

        # Then write voltages_w calculated for previous time step
        self.voltages_w = voltages_w
        result_write = hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])
        if result_write < 0:
            print_possible_error(result_write)

        return np.asarray(self.currents_r), \
            np.asarray(self.encoder_r_buffer), \
            np.asarray(self.other_r_buffer)

    def _set_led(self,
                np.ndarray[qt.t_double, ndim=1, mode='c'] led_w not None):
        self.led_w_buffer = led_w
        result = hil.hil_write_other(
            self.board,
            &self.led_w_channels[0],
            self.num_led_w_channels,
            &self.led_w_buffer[0])
        if result < 0:
            print_possible_error(result)


cdef class QuanserAero(QuanserWrapper):
    def __cinit__(self):
        board_type = b'quanser_aero_usb'
        board_identifier = b'0'
        result = hil.hil_open(board_type, board_identifier, &self.board)
        print_possible_error(result)
        if result < 0:
            raise IOError('Board could not be opened.')

    def __init__(self, frequency=100):
        analog_r_channels = [0, 1]
        analog_w_channels = [0, 1]
        digital_w_channels = [0, 1]
        encoder_r_channels = [0, 1, 2, 3]
        other_r_channels = [3000, 3001, 3002, 4000, 4001, 4002, 14000, \
            14001, 14002, 14003]
        led_w_channels = [11000, 11001, 11002]

        super(QuanserAero, self).__init__(
            safe_operating_voltage=18.0,
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)


cdef class QubeServo2(QuanserWrapper):
    def __cinit__(self):
        board_type = b'qube_servo2_usb'
        board_identifier = b'0'
        result = hil.hil_open(board_type, board_identifier, &self.board)
        print_possible_error(result)
        if result < 0:
            raise IOError('Board could not be opened.')

    def __init__(self, frequency=100):
        analog_r_channels = [0]
        analog_w_channels = [0]
        digital_w_channels = [0]
        encoder_r_channels = [0, 1]
        other_r_channels = [14000]
        led_w_channels = [11000, 11001, 11002]

        super(QubeServo2, self).__init__(
            safe_operating_voltage=18.0,
            analog_r_channels=analog_r_channels,
            analog_w_channels=analog_w_channels,
            digital_w_channels=digital_w_channels,
            encoder_r_channels=encoder_r_channels,
            other_r_channels=other_r_channels,
            led_w_channels=led_w_channels,
            frequency=frequency)
