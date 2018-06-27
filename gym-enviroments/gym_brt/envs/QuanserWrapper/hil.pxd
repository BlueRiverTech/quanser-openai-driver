include "quanser_types.pxd"

# Needed functions:
    #hil_open
    #hil_set_encoder_counts
    #hil_task_create_encoder_reader
    #hil_task_start
    #hil_task_read_encoder
    #hil_write_analog
    #hil_task_stop
    #hil_task_delete
    #hil_close

# From the following header files:
    #include "hil.h"
    #include "quanser_signal.h"
    #include "quanser_messages.h"
    #include "quanser_thread.h"


cdef extern from "/opt/quanser/hil_sdk/include/hil.h":
    ctypedef struct t_card:
        pass
    ctypedef struct t_task:
        pass
    ctypedef struct t_monitor:
        pass
    cdef enum t_clock: # The original c code also defines tag_clock here
        SYSTEM_CLOCK_4    = -4
        SYSTEM_CLOCK_3    = -3
        SYSTEM_CLOCK_2    = -2
        SYSTEM_CLOCK_1    = -1
        HARDWARE_CLOCK_0  = 0
        HARDWARE_CLOCK_1  = 1
        HARDWARE_CLOCK_2  = 2
        HARDWARE_CLOCK_3  = 3
        HARDWARE_CLOCK_4  = 4
        HARDWARE_CLOCK_5  = 5
        HARDWARE_CLOCK_6  = 6
        HARDWARE_CLOCK_7  = 7
        HARDWARE_CLOCK_8  = 8
        HARDWARE_CLOCK_9  = 9
        HARDWARE_CLOCK_10 = 10
        HARDWARE_CLOCK_11 = 11
        HARDWARE_CLOCK_12 = 12
        HARDWARE_CLOCK_13 = 13
        HARDWARE_CLOCK_14 = 14
        HARDWARE_CLOCK_15 = 15
        HARDWARE_CLOCK_16 = 16
        HARDWARE_CLOCK_17 = 17
        HARDWARE_CLOCK_18 = 18
        HARDWARE_CLOCK_19 = 19

    t_error hil_open(
        const char * card_type,
        const char * card_identifier,
        t_card * card)
    t_error hil_close(t_card card)

    # For debugging
    t_error hil_close_all() # Sometimes the board doesn't safely close

    # Synchronous reads/writes
    t_error hil_set_encoder_counts(
        t_card card,
        const t_uint32 encoder_channels[],
        t_uint32 num_channels,
        const t_int32 buffer[])
    t_error hil_write_analog(
        t_card card,
        const t_uint32 analog_channels[],
        t_uint32 num_channels,
        const t_double buffer[])
    t_error hil_write_digital(
        t_card card,
        const t_uint32 digital_lines[],
        t_uint32 num_lines,
        const t_boolean buffer[])
    t_error hil_write_other(
        t_card card,
        const t_uint32 other_channels[],
        t_uint32 num_channels,
        const t_double buffer[])

    t_error hil_read_analog(
        t_card card,
        const t_uint32 analog_channels[],
        t_uint32 num_channels,
        t_double buffer[])
    t_error hil_read_encoder(
        t_card card,
        const t_uint32 encoder_channels[],
        t_uint32 num_channels,
        t_int32 buffer[])
    t_error hil_read_digital(
        t_card card,
        const t_uint32 digital_lines[],
        t_uint32 num_lines,
        t_boolean buffer[])
    t_error hil_read_other(
        t_card card,
        const t_uint32 other_channels[],
        t_uint32 num_channels,
        t_double buffer[])
    t_error hil_read(
        t_card card, 
        const t_uint32 analog_channels[], t_uint32 num_analog_channels, 
        const t_uint32 encoder_channels[], t_uint32 num_encoder_channels, 
        const t_uint32 digital_lines[], t_uint32 num_digital_lines, 
        const t_uint32 other_channels[], t_uint32 num_other_channels, 
        t_double analog_buffer[],
        t_int32 encoder_buffer[],
        t_boolean digital_buffer[],
        t_double other_buffer[])

    # Async
    t_error hil_task_create_reader_writer(
        t_card card, t_uint32 samples_in_buffer,
        const t_uint32 analog_input_channels[],
        t_uint32 num_analog_input_channels,
        const t_uint32 encoder_input_channels[],
        t_uint32 num_encoder_input_channels,
        const t_uint32 digital_input_lines[],
        t_uint32 num_digital_input_lines,
        const t_uint32 other_input_channels[],
        t_uint32 num_other_input_channels,

        const t_uint32 analog_output_channels[],
        t_uint32 num_analog_output_channels,
        const t_uint32 pwm_output_channels[],
        t_uint32 num_pwm_output_channels,
        const t_uint32 digital_output_lines[],
        t_uint32 num_digital_output_lines,
        const t_uint32 other_output_channels[],
        t_uint32 num_other_output_channels,
        t_task *task)
    t_error hil_task_create_reader(
        t_card card, t_uint32 samples_in_buffer,
        const t_uint32 analog_channels[],
        t_uint32 num_analog_channels,
        const t_uint32 encoder_channels[],
        t_uint32 num_encoder_channels,
        const t_uint32 digital_lines[],
        t_uint32 num_digital_lines, 
        const t_uint32 other_channels[],
        t_uint32 num_other_channels,
        t_task *task)

    t_error hil_task_start(
        t_task task,
        t_clock clock,
        t_double frequency,
        t_uint32 num_samples)
    t_error hil_task_flush(t_task task)
    t_error hil_task_stop(t_task task)
    t_error hil_task_delete(t_task task)

    t_error hil_task_read_write(
        t_task task, t_uint32 num_samples,
        t_double analog_input_buffer[],
        t_int32 encoder_input_buffer[],
        t_boolean digital_input_buffer[],
        t_double other_input_buffer[],

        const t_double analog_output_buffer[],
        const t_double pwm_output_buffer[],
        const t_boolean digital_output_buffer[],
        const t_double other_output_buffer[])
    t_error hil_task_write(
        t_task task, t_uint32 num_samples,
        const t_double analog_buffer[],
        const t_double pwm_buffer[],
        const t_boolean digital_buffer[],
        const t_double other_buffer[])
    t_error hil_task_read(
        t_task task, t_uint32 num_samples,
        t_double analog_buffer[],
        t_int32 encoder_buffer[],
        t_boolean digital_buffer[],
        t_double other_buffer[])
