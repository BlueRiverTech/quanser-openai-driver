cimport numpy as np

cdef extern from "/opt/quanser/hil_sdk/include/quanser_types.h":
    ctypedef int t_error

    ctypedef np.npy_int8    t_boolean # must always be 8 bits
    ctypedef np.npy_int8    t_byte    # must always be 8 bits
    ctypedef np.npy_uint8   t_ubyte   # must always be 8 bits
    ctypedef np.npy_int16   t_short   # must always be 16 bits
    ctypedef np.npy_uint16  t_ushort  # must always be 16 bits
    ctypedef np.npy_int32   t_int     # must always be 32 bits
    ctypedef np.npy_uint32  t_uint    # must always be 32 bits
    ctypedef np.npy_float64 t_double  # must always be 64 bits
    ctypedef np.npy_int64   t_long    # must always be 64 bits

    ctypedef np.npy_int8    t_int8
    ctypedef np.npy_uint8   t_uint8
    ctypedef np.npy_int16   t_int16
    ctypedef np.npy_uint16  t_uint16
    ctypedef np.npy_int32   t_int32
    ctypedef np.npy_uint32  t_uint32
