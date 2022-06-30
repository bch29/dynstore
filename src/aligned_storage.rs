const CHUNK_SIZE: usize = 2 << 20; // 2 MB
const MAX_ALIGN: usize = 1024;

pub struct AlignedStorage {
    chunk_capacity: usize,
    chunks: Vec<Chunk>,
}

struct Chunk {
    len: usize,
    capacity: usize,
    data: *mut u8,
}

fn layout(capacity: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(capacity, MAX_ALIGN).unwrap()
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.data as *mut u8, layout(self.capacity)) }
    }
}

impl AlignedStorage {
    pub fn new() -> Self {
        Self::with_chunk_size(CHUNK_SIZE)
    }

    pub fn with_chunk_size(chunk_capacity: usize) -> Self {
        Self {
            chunks: vec![],
            chunk_capacity,
        }
    }

    fn alloc_chunk(capacity: usize) -> Chunk {
        let data = unsafe { std::alloc::alloc(layout(capacity)) };

        Chunk {
            len: 0,
            capacity,
            data,
        }
    }

    pub fn reserve<T>(&mut self) -> (*mut T, (u32, u32)) {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();

        if align > MAX_ALIGN {
            panic!(
                "aligment requirement for type is {} which is greater than max alignment of {}",
                align, MAX_ALIGN
            );
        }

        if self.chunks.is_empty() {
            self.chunks
                .push(Self::alloc_chunk(size.max(self.chunk_capacity)));
        }

        let mut current_chunk = self.chunks.last_mut().unwrap();

        let mut offset = (current_chunk.len / align) * align;
        if offset < current_chunk.len {
            offset += align;
        }

        if offset + size > current_chunk.capacity {
            self.chunks
                .push(Self::alloc_chunk(size.max(self.chunk_capacity)));
            offset = 0;
            current_chunk = self.chunks.last_mut().unwrap();
        }

        let addr = unsafe { current_chunk.data.offset(offset as isize) as *mut T };

        current_chunk.len = offset + size;

        (addr, ((self.chunks.len() - 1) as u32, offset as u32))
    }

    #[cfg(test)]
    pub fn allocate<T>(&mut self, value: T) -> (*mut T, (u32, u32)) {
        let (addr, offset) = self.reserve::<T>();

        unsafe {
            std::ptr::write(addr, value);
        };

        (addr, offset)
    }

    pub fn offset_mut(&mut self, offset: (u32, u32)) -> *mut () {
        let (chunk_ix, offset) = offset;
        assert!((offset as usize) < self.chunk_capacity);
        let chunk = &mut self.chunks[chunk_ix as usize];
        unsafe { chunk.data.offset(offset as isize) as *mut () }
    }

    pub fn offset(&self, offset: (u32, u32)) -> *const () {
        let (chunk_ix, offset) = offset;
        assert!((offset as usize) < self.chunk_capacity);
        let chunk = &self.chunks[chunk_ix as usize];
        unsafe { chunk.data.offset(offset as isize) as *const () }
    }
}

impl Default for AlignedStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::AlignedStorage;

    #[test]
    fn test_aligned_storage() {
        let mut storage = AlignedStorage::new();

        let (pa, (oa1, _)) = storage.allocate(10i32);
        let (pb, (ob1, _)) = storage.allocate("foo".to_string());
        let (pc, (oc1, _)) = storage.allocate(vec![-1i32, 2, -10]);

        assert_eq!(oa1, 0);
        assert_eq!(ob1, 0);
        assert_eq!(oc1, 0);

        let a: &i32 = unsafe { &*pa };
        assert_eq!(a, &10);

        let b: &String = unsafe { &*pb };
        assert_eq!(b.as_str(), "foo");

        let c: &Vec<i32> = unsafe { &*pc };
        assert_eq!(c.as_slice(), &[-1, 2, -10]);

        // manual drops to satisfy miri's leak checker
        unsafe { std::ptr::drop_in_place(pb as *mut String) };
        unsafe { std::ptr::drop_in_place(pc as *mut Vec<i32>) };
    }

    #[test]
    fn test_big_aligned_storage() {
        const CAPACITY: usize = 1 << 10;

        let mut storage = AlignedStorage::with_chunk_size(CAPACITY);

        let size = std::mem::size_of::<String>();
        let data_per_chunk = CAPACITY / size;

        let pointers = (0..(CAPACITY * 2 / size - 1)).into_iter().map(|i| {
            let (ptr, (chunk, offset)) = storage.reserve::<String>();

            if i < data_per_chunk {
                assert_eq!(chunk, 0);
                assert_eq!(offset as usize, i * size)
            } else {
                assert_eq!(chunk, 1);
                assert_eq!(offset as usize, (i - data_per_chunk) * size)
            }

            unsafe {
                std::ptr::write_volatile(ptr, i.to_string());
            }
            ptr
        });

        for (i, ptr) in pointers.enumerate() {
            // NB the read also causes the string to be dropped, which is fine because the
            // AlignedStorage doesn't drop anything written into it on its own
            let value = unsafe { std::ptr::read_volatile(ptr) };
            assert_eq!(value, i.to_string());
        }
    }
}
