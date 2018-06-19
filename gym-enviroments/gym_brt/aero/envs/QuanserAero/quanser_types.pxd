

cdef extern from "/opt/quanser/hil_sdk/include/quanser_types.h":
    ctypedef int t_error

    ### TODO HANDLE THIS STUFF PROPERLY!!! ###
    # My system:
        # long long is 64 bits
        # int is 32 bits
        # short is 16 bits
        # char is 8 bits

        # char=1 bytes
        # short=2 bytes
        # int=4 bytes
        # long=8 bytes
        # long long=8 bytes
        # float=4 bytes
        # double=8 bytes
        # long double=16 bytes

    ctypedef char           t_boolean    # must always be 8 bits
    ctypedef signed char    t_byte       # must always be 8 bits
    ctypedef unsigned char  t_ubyte      # must always be 8 bits
    ctypedef signed short   t_short      # must always be 16 bits
    ctypedef unsigned short t_ushort     # must always be 16 bits
    ctypedef signed int     t_int        # must always be 32 bits
    ctypedef unsigned int   t_uint       # must always be 32 bits
    ctypedef double         t_double     # must always be 8 bytes (64 bits)

    ctypedef t_byte         t_int8
    ctypedef t_ubyte        t_uint8
    ctypedef t_short        t_int16
    ctypedef t_ushort       t_uint16
    ctypedef t_int          t_int32
    ctypedef t_uint         t_uint32
