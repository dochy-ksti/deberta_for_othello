#![allow(dead_code)]

use std::fmt::Display;
pub struct CArray<T> {
    pub size: Vec<usize>,
    pub array: Vec<T>,
}

impl<T> CArray<T>
where
    T: Default + Clone + Copy + Display,
{
    pub fn new1(size: usize) -> Self {
        Self {
            size: vec![size],
            array: vec![T::default(); size],
        }
    }

    pub fn new2(size_x: usize, size_y: usize) -> Self {
        let size = vec![size_x, size_y];
        let array = vec![T::default(); size_x * size_y];
        Self { size, array }
    }

    pub fn new3(size_x: usize, size_y: usize, size_z: usize) -> Self {
        let size = vec![size_x, size_y, size_z];
        let array = vec![T::default(); size_x * size_y * size_z];
        Self { size, array }
    }

    pub fn dimension(&self) -> usize {
        self.size.len()
    }

    pub fn size(&self) -> &[usize] {
        &self.size
    }

    pub fn size0(&self) -> usize {
        self.size[0]
    }

    pub fn size1(&self) -> usize {
        self.size[1]
    }

    pub fn size2(&self) -> usize {
        self.size[2]
    }

    pub fn get1(&self, x: usize) -> T {
        debug_assert!(self.size.len() == 1);
        debug_assert!(x < self.size[0]);
        self.array[x]
    }

    pub fn set1(&mut self, x: usize, v: T) {
        debug_assert!(self.size.len() == 1);
        debug_assert!(x < self.size[0]);
        self.array[x] = v;
    }

    pub fn ref2(&self, x: usize) -> &[T] {
        debug_assert!(self.size.len() == 2);
        debug_assert!(x < self.size[0]);
        let r = &self.array[x * self.size[1]..];
        &r[..self.size[1]]
    }

    pub fn get2(&self, x: usize, y: usize) -> T {
        debug_assert!(self.size.len() == 2);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        self.array[x * self.size[1] + y]
    }

    pub fn ref_mut2(&mut self, x: usize) -> &mut [T] {
        debug_assert!(self.size.len() == 2);
        debug_assert!(x < self.size[0]);
        let r = &mut self.array[x * self.size[1]..];
        &mut r[..self.size[1]]
    }

    pub fn set2(&mut self, x: usize, y: usize, v: T) {
        debug_assert!(self.size.len() == 2);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        self.array[x * self.size[1] + y] = v;
    }

    pub fn ref3_1(&self, x: usize) -> &[T] {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        let r = &self.array[x * self.size[1] * self.size[2]..];
        &r[..self.size[1] * self.size[2]]
    }

    pub fn ref3_2(&self, x: usize, y: usize) -> &[T] {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        let r = &self.array[x * self.size[1] * self.size[2] + y * self.size[2]..];
        &r[..self.size[2]]
    }

    pub fn get3(&self, x: usize, y: usize, z: usize) -> T {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        debug_assert!(z < self.size[2]);
        self.array[x * self.size[1] * self.size[2] + y * self.size[2] + z]
    }

    pub fn ref_mut3_1(&mut self, x: usize) -> &mut [T] {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        let r = &mut self.array[x * self.size[1] * self.size[2]..];
        &mut r[..self.size[1] * self.size[2]]
    }

    pub fn ref_mut3_2(&mut self, x: usize, y: usize) -> &mut [T] {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        let r = &mut self.array[x * self.size[1] * self.size[2] + y * self.size[2]..];
        &mut r[..self.size[2]]
    }

    pub fn set3(&mut self, x: usize, y: usize, z: usize, v: T) {
        debug_assert!(self.size.len() == 3);
        debug_assert!(x < self.size[0]);
        debug_assert!(y < self.size[1]);
        debug_assert!(z < self.size[2]);
        self.array[x * self.size[1] * self.size[2] + y * self.size[2] + z] = v;
    }

    pub fn as_ref(&self) -> &[T] {
        &self.array
    }

    pub fn as_mut(&mut self) -> &mut [T] {
        &mut self.array
    }

    pub fn as_ptr(&self) -> *const T {
        self.array.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.array.as_mut_ptr()
    }

    pub fn _to_string(&self) -> String {
        let mut r = String::new();
        Self::_to_str(&self.size, self.array.as_slice(), &mut r);
        r
    }

    fn _to_str(size: &[usize], contents: &[T], r: &mut String) {
        if size.len() == 1 {
            for c in contents {
                r.push_str(&format!("{:4.1} ", c))
            }
        } else {
            let stride = size[1..].iter().fold(1, |a,b| a * *b);
            for i in 0..size[0] {
                let part = &contents[i * stride..];
                let part = &part[..stride];
                Self::_to_str(&size[1..], part, r);
                r.push_str("\n");
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn create_carray1(size_x: usize) -> *mut CArray<f32> {
    let b = Box::new(CArray::new1(size_x));
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn create_carray2(size_x: usize, size_y: usize) -> *mut CArray<f32> {
    let b = Box::new(CArray::new2(size_x, size_y));
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn create_carray3(size_x: usize, size_y: usize, size_z: usize) -> *mut CArray<f32> {
    let b = Box::new(CArray::new3(size_x, size_y, size_z));
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn destroy_carray(p: *mut CArray<f32>) {
    unsafe {
        let _ = Box::from_raw(p);
    }
}

#[no_mangle]
pub extern "C" fn carray_get1(p: *const CArray<f32>, x: usize) -> f32 {
    unsafe { (*p).get1(x) }
}

#[no_mangle]
pub extern "C" fn carray_set1(p: *mut CArray<f32>, x: usize, v: f32) {
    unsafe { (*p).set1(x, v) }
}

#[no_mangle]
pub extern "C" fn carray_get2(p: *const CArray<f32>, x: usize, y: usize) -> f32 {
    unsafe { (*p).get2(x, y) }
}

#[no_mangle]
pub extern "C" fn carray_set2(p: *mut CArray<f32>, x: usize, y: usize, v: f32) {
    unsafe { (*p).set2(x, y, v) }
}

#[no_mangle]
pub extern "C" fn carray_get3(p: *const CArray<f32>, x: usize, y: usize, z: usize) -> f32 {
    unsafe { (*p).get3(x, y, z) }
}

#[no_mangle]
pub extern "C" fn carray_set3(p: *mut CArray<f32>, x: usize, y: usize, z: usize, v: f32) {
    unsafe { (*p).set3(x, y, z, v) }
}

#[no_mangle]
pub extern "C" fn carray_as_ptr(p: *mut CArray<f32>) -> *mut f32 {
    unsafe { (*p).as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn carray_dimension(p: *mut CArray<f32>) -> usize {
    unsafe { (*p).dimension() }
}

#[no_mangle]
pub extern "C" fn carray_size0(p: *mut CArray<f32>) -> usize {
    unsafe { (*p).size0() }
}
#[no_mangle]
pub extern "C" fn carray_size1(p: *mut CArray<f32>) -> usize {
    unsafe { (*p).size1() }
}
#[no_mangle]
pub extern "C" fn carray_size2(p: *mut CArray<f32>) -> usize {
    unsafe { (*p).size2() }
}

#[no_mangle]
pub extern "C" fn carray_as_ptr2(p: *mut CArray<f32>, x: usize) -> *mut f32 {
    unsafe { (*p).ref_mut2(x).as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn carray_as_ptr3_1(p: *mut CArray<f32>, x: usize) -> *mut f32 {
    unsafe { (*p).ref_mut3_1(x).as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn carray_as_ptr3_2(p: *mut CArray<f32>, x: usize, y: usize) -> *mut f32 {
    unsafe { (*p).ref_mut3_2(x, y).as_mut_ptr() }
}
