# On linux, __off_t is _OFF_T_SIZE,
# which is __SLONGWORD_TYPE,
# which is long int (in bits/types.h)
cdef extern from "fcntl.h":
    enum: POSIX_FADV_DONTNEED
    int posix_fadvise (int __fd, long int __offset, long int __len,
                       int __advise)
