import collections

cimport posix.unistd
cimport posix.fcntl

cdef void c_dontneed(char* filename):
    cdef int fd
    fd = posix.fcntl.open(filename, posix.fcntl.O_RDONLY)
    if fd == -1:
        raise IOError("File %s does not exists" % filename)
    posix.unistd.fdatasync(fd)
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    posix.unistd.close(fd)

def dontneed(filename):
    """Call this function to invalidate any cache related
    to the file."""
    if isinstance(filename, str):
        c_dontneed(filename)
    elif isinstance(filename, collections.Iterable):
        for fname in filename:
            c_dontneed(fname)
