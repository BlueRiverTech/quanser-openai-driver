from __future__ import print_function

cimport quanser_types as qt
cimport numpy as np
cimport hil

from gym_brt.envs.QuanserWrapper.helpers.error_codes import print_possible_error

from threading import Thread, Lock
import numpy as np
import time


cdef class QuanserWrapper:
    cdef hil.t_card  board
    cdef hil.t_error result
    cdef hil.t_task  task

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

    cdef qt.t_double frequency, period

    cdef float _last_read_time
    cdef bint _task_started, _new_state_read
    cdef object _bg_thread, _lock
    cdef int _num_samples_read_since_action

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

        self._lock = Lock()
        self.frequency = frequency
        self._task_started = False
        self._new_state_read = False
        self._num_samples_read_since_action = 0
        self._bg_thread = Thread(target=self.run_reader_writer, args=())
        self._last_read_time = 0

    def __enter__(self):
        """
        Start the hardware in a deterministic way (all motors, encoders, etc
        at 0)
        """
        # Create a memoryview for currents
        self.currents_r = np.zeros(
            self.num_analog_r_channels, dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for -ometers
        self.other_r_buffer = np.zeros(
            self.num_other_r_channels, dtype=np.float64) # t_double is 64 bits

        # Create a memoryview for leds
        self.led_w_buffer = np.zeros(
            self.num_led_w_channels, dtype=np.float64) # t_double is 64 bits

        # Set all motor voltages_w to 0
        self.voltages_w = np.zeros(
            self.num_analog_w_channels, dtype=np.float64) # t_double is 64 bits
        result = hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])
        print_possible_error(result)

        # Set the encoder encoder_r_buffer to 0
        self.encoder_r_buffer = np.zeros(
            self.num_encoder_r_channels, dtype=np.int32) # t_int32 is 32 bits
        result = hil.hil_set_encoder_counts(
            self.board,
            &self.encoder_r_channels[0],
            self.num_encoder_r_channels,
            &self.encoder_r_buffer[0])
        print_possible_error(result)

        # Enables_r all the motors
        self.enables_r = np.ones(
            self.num_digital_w_channels, dtype=np.int8) # t_bool is char 8 bits
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
            self.num_analog_w_channels, dtype=np.float64) # t_double is 64 bits
        hil.hil_write_analog(
            self.board,
            &self.analog_w_channels[0],
            self.num_analog_w_channels,
            &self.voltages_w[0])

        # Disable all the motors
        self.enables_r = np.zeros(
            self.num_digital_w_channels, dtype=np.int8) # t_bool is char 8 bits
        hil.hil_write_digital(
            self.board,
            &self.digital_w_channels[0],
            self.num_digital_w_channels,
            &self.enables_r[0])

    def _create_task(self):
        """Start a task reads and writes at fixed intervals"""
        result =  hil.hil_task_create_reader(
            self.board,
            1,  # The size of the internal buffer (making this >> 1 
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
        if result < 0:
            raise ValueError("Could not start hil task")

        self._task_started = True
        self._bg_thread.start()

    def _stop_task(self):
        if self._task_started:
            self._task_started = False
            self._bg_thread.join()
            hil.hil_task_flush(self.task)
            hil.hil_task_stop(self.task)
            hil.hil_task_delete(self.task)

    def run_reader_writer(self):
        """Helper function to pass as a python callable to `Thread`"""
        self._run_reader_writer()

    cdef _run_reader_writer(self):
        """Run background thread that continously updates QuanserWrapper's
        internal buffers with the newest state at a sample instant, and writes
        the current action buffer to the board.
        """
        cdef hil.t_error samples_read, result_write
        cdef qt.t_double[::] temp_currents_r = np.empty_like(self.currents_r)
        cdef qt.t_int32[::] temp_encoder_r_buffer = np.empty_like(
            self.encoder_r_buffer)
        cdef qt.t_double[::] temp_other_r_buffer = np.empty_like(
            self.other_r_buffer)

        while self._task_started:
            # First read using task_read (blocking call that enforces timing)
            samples_read = hil.hil_task_read(
                self.task,
                1, # Number of samples to read
                &temp_currents_r[0],
                &temp_encoder_r_buffer[0],
                NULL,
                &temp_other_r_buffer[0])
            if samples_read < 0:
                print_possible_error(samples_read)

            with self._lock:
                # Copy the temp state buffers into the quanser wrapper buffers
                self.currents_r = temp_currents_r
                self.encoder_r_buffer = temp_encoder_r_buffer
                self.other_r_buffer = temp_other_r_buffer

                # Then write voltages_w calculated for previous time step
                result_write = hil.hil_write_analog(
                    self.board,
                    &self.analog_w_channels[0],
                    self.num_analog_w_channels,
                    &self.voltages_w[0])
                if result_write < 0:
                    print_possible_error(result_write)

                self._new_state_read = True
                self._num_samples_read_since_action += 1

            time.sleep(0.1 / self.frequency)

    def action(self, voltages_w):
        """Make sure you get safe data!"""    
        # If it's the first time running action, then start the background r/w 
        # task
        if not self._task_started:
            self._create_task()

        if isinstance(voltages_w, list):
            voltages_w = np.array(voltages_w, dtype=np.float64)
        assert isinstance(voltages_w, np.ndarray)
        assert voltages_w.shape == (self.num_analog_w_channels,)
        assert voltages_w.dtype == np.float64
        for i in range(self.num_analog_w_channels):
            assert -25.0 <= voltages_w[i] <= 25.0 # Operating range

        self._action(voltages_w)
        self._action(voltages_w)
        self._action(voltages_w)
        return self._action(voltages_w)

    def _action(self,
                np.ndarray[qt.t_double, ndim=1, mode="c"] voltages_w not None):
        """Perform actions on the device (voltages_w must always be ndarray!)"""

        with self._lock:
            # Print warning if buffer read has been missed
            if self._num_samples_read_since_action > 1:
                print("Warning:", self._num_samples_read_since_action - 1,
                      "samples have been missed since last env step")

            # Update the action in the quanser wrapper buffer
            self.voltages_w = voltages_w.copy()

        while True:
            # Make sure to get the most recent state from the background reader
            time.sleep(0.1 / self.frequency)
            with self._lock:
                if self._new_state_read:
                    currents_r = np.asarray(self.currents_r).copy()
                    encoder_r_buffer = np.asarray(self.encoder_r_buffer).copy()
                    other_r_buffer = np.asarray(self.other_r_buffer).copy()
                    self._num_samples_read_since_action = 0
                    self._new_state_read = False
                    break
        return currents_r, encoder_r_buffer, other_r_buffer


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
        other_r_channels = [3000, 3001, 3002, 4000, 4001, 4002, 14000, 14001, \
                             14002, 14003]
        led_w_channels = [11000, 11001, 11002]

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
        other_r_channels = [14000]
        led_w_channels = [11000, 11001, 11002]

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

