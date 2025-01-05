from ctypes import c_void_p, c_size_t, c_float, POINTER, CDLL
import ctypes
from typing import Sequence
from numpy.typing import NDArray
from numpy import float32
import numpy as np


class CArray:
    def __init__(self, lib: CDLL, p: c_void_p):
        self.lib = lib
        self.p = p

    def __del__(self):
        self.lib.destroy_carray(self.p)

    def to_numpy(self) -> NDArray[float32]:
        p = self.lib.carray_as_ptr(self.p)
        return np.ctypeslib.as_array(p, shape=self.shape()).copy()

    @staticmethod
    def from_numpy(lib: CDLL, array: NDArray[float32]) -> "CArray":
        carray: CArray = CArray.__create_carray_from_shape(lib, array.shape)
        ptr = lib.carray_as_ptr(carray.p)

        ctypes.memmove(ptr, array.ctypes.data, array.size *
                       ctypes.sizeof(ctypes.c_float))
        return carray

    @staticmethod
    def __create_carray_from_shape(lib: CDLL, shape: Sequence[int]) -> "CArray":
        if len(shape) == 1:
            return CArray(lib, lib.create_carray1(shape[0]))
        elif len(shape) == 2:
            return CArray(lib, lib.create_carray2(shape[0], shape[1]))
        elif len(shape) == 3:
            return CArray(lib, lib.create_carray3(shape[0], shape[1], shape[2]))
        else:
            raise ValueError("CArray's dimension must be 1,2 or 3")

    def shape(self) -> list[int]:
        dimension = self.lib.carray_dimension(self.p)
        if dimension == 1:
            return [self.lib.carray_size0(self.p)]
        elif dimension == 2:
            return [self.lib.carray_size0(self.p), self.lib.carray_size1(self.p)]
        elif dimension == 3:
            return [self.lib.carray_size0(self.p), self.lib.carray_size1(self.p), self.lib.carray_size2(self.p)]
        else:
            raise ValueError("CArray's dimension must be 1,2 or 3")


def define_carray_funcs(lib: CDLL):
    lib.create_carray1.argtypes = [c_size_t]
    lib.create_carray1.restype = POINTER(c_void_p)
    lib.create_carray2.argtypes = [c_size_t, c_size_t]
    lib.create_carray2.restype = POINTER(c_void_p)
    lib.create_carray3.argtypes = [c_size_t, c_size_t, c_size_t]
    lib.create_carray2.restype = POINTER(c_void_p)
    lib.destroy_carray.argtypes = [POINTER(c_void_p)]

    lib.carray_get1.argtypes = [POINTER(
        c_void_p), c_size_t]
    lib.carray_get1.restype = c_float
    lib.carray_set1.argtypes = [POINTER(
        c_void_p), c_size_t, c_float]
    lib.carray_get2.argtypes = [POINTER(
        c_void_p), c_size_t, c_size_t]
    lib.carray_get2.restype = c_float
    lib.carray_set2.argtypes = [POINTER(
        c_void_p), c_size_t, c_size_t, c_float]
    lib.carray_get3.argtypes = [POINTER(
        c_void_p), c_size_t, c_size_t, c_size_t]
    lib.carray_get3.restype = c_float
    lib.carray_set3.argtypes = [POINTER(
        c_void_p), c_size_t, c_size_t, c_size_t, c_float]
    lib.carray_as_ptr.argtypes = [POINTER(c_void_p)]
    lib.carray_as_ptr.restype = POINTER(c_float)
    lib.carray_dimension.argtypes = [POINTER(c_void_p)]
    lib.carray_dimension.restype = c_size_t
    lib.carray_size0.argtypes = [POINTER(c_void_p)]
    lib.carray_size0.restype = c_size_t
    lib.carray_size1.argtypes = [POINTER(c_void_p)]
    lib.carray_size1.restype = c_size_t
    lib.carray_size2.argtypes = [POINTER(c_void_p)]
    lib.carray_size2.restype = c_size_t
    lib.carray_as_ptr2.argtypes = [POINTER(c_void_p), c_size_t]
    lib.carray_as_ptr2.restype = POINTER(c_float)
    lib.carray_as_ptr3_1.argtypes = [POINTER(c_void_p), c_size_t]
    lib.carray_as_ptr3_1.restype = POINTER(c_float)
    lib.carray_as_ptr3_2.argtypes = [POINTER(c_void_p), c_size_t, c_size_t]
    lib.carray_as_ptr3_2.restype = POINTER(c_float)
