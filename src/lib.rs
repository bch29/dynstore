//! Provides the [`ObjectStore`] type, a container which can store arbitrary objects as long as
//! they implement [`Castable`].
//!
//! Provides lifetime-free handles which can later be safely used to access
//! references to the stored objects.  Supports casting handles to trait objects for any trait
//! implemented by a stored object and registered in its [`Castable`] instance.  The internal data
//! layout efficiently packs objects, storing minimal metadata for type casting and searches
//! separately.
//!
//! # Variants
//!
//! Supports variants dependending on the kind of types you need to store:
//! - `Send` and `Sync`: [`ObjectStore<tag::SendSync>`]
//! - `Send` but not `Sync`: [`ObjectStore<tag::Send>`]
//! - neither `Send` nor `Sync`: [`ObjectStore<tag::ThreadLocal>`]
//!
//! # Buffers
//!
//! The [`ObjectStore`] contains multiple conceptual "buffers" keyed by a [`u32`]. When you push an
//! object to the store, you have to choose which buffer index it gets assigned to. In the simplest
//! usage, only one buffer (index 0) can be used. However, making use of multiple buffers can have
//! performance benefits:
//! - Objects pushed to the same buffer are stored near to each other in memory.
//! - [`ObjectStore::find`] operates on a single buffer so the buffer index can be treated as a
//! primary key to speed up searching.
//!
//! # Common Operations
//!
//! - Use [`ObjectStore::push`] to push a new object to the store.
//! - Use [`ObjectStore::get`] to convert a [`Handle`] into a reference.
//! - Use [`ObjectStore::get_mut`] to convert a [`Handle`] into a mutable reference.
//! - Use [`ObjectStore::cast`] to safely cast a [`Handle`] into a handle of a different type.
//! - Use [`ObjectStore::find`] to find objects castable to a given type.
//!
//! # Caveats
//!
//! The motivating use case is to store objects that live for the entire lifetime of the program.
//! Thus, once objects are pushed to the [`ObjectStore`], they are not dropped until the whole
//! store is dropped. Due to the way handles are implemented, there is also a limitation that at
//! most 2^32 [`ObjectStore`] instances can be created over the whole life of a program. This might
//! limit some use cases as a temporary arena allocator.
//!
//! Pushing objects into the store is not fast enough to do in a hot loop. It involves a hash
//! lookup and a few indirections in the common case, and some allocations and linear searches
//! whenever an object of a previously-unseen type is pushed. The store is primarily designed for
//! fast access, searches and casts, not fast allocation.
//!
//! Internally, memory is segmented into allocations of size 2MB. This has three main consequences
//! that users need to be aware of:
//! - Any attempt to push an individual object larger than 2MB to an [`ObjectStore`] will panic.
//! - If object sizes are a significant fraction of 2MB, memory can be wasted at the end of each
//! segment.
//! - At least 2MB will be allocated per buffer index used. If 1000 different buffer indexes are
//! used, this is 2GB which is quite significant. It is therefore not efficient to store a few
//! small objects across many buffers.

mod aligned_storage;
mod castable;

use crate::aligned_storage::AlignedStorage;
#[cfg(feature = "inventory")]
pub use crate::castable::RegisterCast;
pub use crate::castable::{assert_implements_castable, Castable, Casts};

use std::any::TypeId;
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

/// Tags for controlling what can be stored in an [`ObjectStore`].
pub mod tag {
    /// Tag for `ObjectStore`s that can store values that are neither [`std::marker::Send`] nor
    /// [`Sync`].
    pub struct ThreadLocal(std::marker::PhantomData<*const ()>);

    /// Tag for `ObjectStore`s that can store values that are [`std::marker::Send`] but not
    /// [`Sync`].
    pub struct Send(std::marker::PhantomData<*const ()>);

    /// Tag for `ObjectStore`s that requires all values to be both [`std::marker::Send`] and
    /// [`Sync`].
    pub struct SendSync;

    // SAFETY: this tag doesn't store any data
    unsafe impl std::marker::Send for Send {}

    /// Converts a tag into a trait bound.
    ///
    /// # Safety
    /// It is unsound to implement `Satisfies<Tag> for T` whenever `Tag` implements a trait that
    /// `T` does not also implement.
    pub unsafe trait Satisfies<Tag> {}
    unsafe impl<T> Satisfies<ThreadLocal> for T {}
    unsafe impl<T> Satisfies<Send> for T where T: std::marker::Send {}
    unsafe impl<T> Satisfies<SendSync> for T where T: std::marker::Send + Sync {}
}

/// Stores arbitrary objects as long as they implement [`Castable`] and [`tag::Satisfies<Tag>`] for
/// the choice of `Tag`.
///
/// See the module documentation for more info.
pub struct ObjectStore<Tag = tag::Send> {
    _marker: std::marker::PhantomData<Tag>,
    inner: ObjectStoreInner,
}

/// A partially initialized object store containing potentially uninitialized objects. Supports
/// reserving slots for data and provides handles. Does not allow access to any contained data
/// until the store is converted to an [`ObjectStore`] using [`UninitObjectStore::try_init`]. This
/// is only possible when every handle has been written to.
pub struct UninitObjectStore<Tag = tag::Send> {
    _marker: std::marker::PhantomData<Tag>,
    inner: ObjectStoreInner,
    unwritten_objects: HashMap<*const (), UninitObjectInfo>,
}

struct ObjectStoreInner {
    id: u32,
    metas: Vec<ObjectMeta>,
    casts: Casts,
    metas_by_type: HashMap<TypeId, u32>,
    metas_by_cast_target: BTreeSet<(TypeId, u32)>,
    object_keys: BTreeSet<ObjectKey>,
    buffers: Vec<AlignedStorage>,
}

/// A handle pointing to an object of type `T`. Can be used along with the `ObjectStore` that
/// provided the handle to get a `&T`. Can be cast into handles for anything that the underlying
/// type is [`Castable`] into.
pub struct Handle<T: ?Sized> {
    store_id: u32,
    meta_ix: u32,
    ptr: *const T,
}

/// A handle pointing to an object of an unspecified type. Can be cast to any type that the real
/// underlying type can be cast to. Can be used to store heterogeneous handles in one place.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct DynamicHandle {
    inner: Handle<()>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
struct ObjectKey {
    buffer_ix: u32,
    meta_ix: u32,
    offset: (u32, u32),
}

/// Provides information about an uninitialized object in an [`UninitObjectStore`].
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct UninitObjectInfo {
    pub buffer_ix: u32,
    meta_ix: u32,
}

struct ObjectMeta {
    name: String,
    type_id: TypeId,
    destroy: fn(*mut ()),
}

// ------------------------------------------------------------------------------

impl ObjectStore<tag::ThreadLocal> {
    /// Instantiates a new [`ObjectStore`] object that can store objects which are `!Send` and
    /// `!Sync`.
    pub fn new_thread_local() -> Self {
        Self::new()
    }
}

impl ObjectStore<tag::Send> {
    /// Instantiates a new [`ObjectStore`] object that can store objects which are `Send` but
    /// `!Sync`.
    pub fn new_send() -> Self {
        Self::new()
    }
}

impl ObjectStore<tag::SendSync> {
    /// Instantiates a new [`ObjectStore`] object that can store objects which are `Send` and `Sync`.
    pub fn new_sync() -> Self {
        Self::new()
    }
}

impl<Tag> ObjectStore<Tag> {
    /// Instantiates a new [`ObjectStore`] object.
    pub fn new() -> Self {
        ObjectStore {
            _marker: std::marker::PhantomData,
            inner: ObjectStoreInner::new(),
        }
    }

    /// Pushes a new object to the given buffer and returns a handle for it. Any [`u32`] can be
    /// chosen as the `buffer_ix` but bear in mind that a large choice of index will cause a large
    /// allocation to happen if smaller buffer indexes have not yet been used.
    pub fn push<T>(&mut self, buffer_ix: u32, item: T) -> Handle<T>
    where
        T: Castable + tag::Satisfies<Tag>,
    {
        self.inner.push(buffer_ix, item)
    }

    /// Iterates over handles for all stored objects in the given buffer that are castable to the
    /// requested type. This is a fast operation implemented as a range lookup in a BTreeSet.
    pub fn find<'a, T>(&'a self, buffer_ix: u32) -> impl Iterator<Item = Handle<T>> + 'a
    where
        T: ?Sized + 'static,
    {
        self.inner.find::<T>(buffer_ix)
    }

    /// Iterates over dynamic handles for all stored objects in the given buffer that are castable
    /// to the requested type. This is a fast operation implemented as a range lookup in a
    /// BTreeSet.
    pub fn find_dynamic<'a>(
        &'a self,
        buffer_ix: u32,
        type_id: TypeId,
    ) -> impl Iterator<Item = DynamicHandle> + 'a {
        self.inner.find_dynamic(buffer_ix, type_id)
    }

    /// Casts an untyped `DynamicHandle` up to a typed `Handle`. Panics if the input handle is not
    /// associated with this `ObjectStore`. Returns `None` if the object that the handle
    /// points to is not castable to the requested type.
    pub fn cast_from_dynamic<T>(&self, handle: DynamicHandle) -> Option<Handle<T>>
    where
        T: ?Sized + 'static,
    {
        self.inner.cast_from_dynamic(handle)
    }

    /// Casts a handle down to an untyped DynamicHandle. Panics if the input handle is not
    /// associated with this `ObjectStore`.
    pub fn cast_to_dynamic<T>(&self, handle: Handle<T>) -> DynamicHandle
    where
        T: ?Sized,
    {
        self.inner.cast_to_dynamic(handle)
    }

    /// Casts a handle to a different type. Panics if the input handle is not associated with this
    /// `ObjectStore`. Returns `None` if the concrete type pointed to by the input handle is
    /// not castable to `U`.
    pub fn cast<T, U>(&self, handle: Handle<T>) -> Option<Handle<U>>
    where
        T: ?Sized + 'static,
        U: ?Sized + 'static,
    {
        self.inner.cast(handle)
    }

    /// Accesses an object reference through a handle. Panics if the handle is not associated
    /// with this `ObjectStore`. This is very cheap: about as cheap as a bounds-checked index into
    /// an array.
    pub fn get<T>(&self, handle: Handle<T>) -> &T
    where
        T: ?Sized + 'static,
    {
        self.inner.get(handle)
    }

    /// Accesses a mutable object reference through a handle. Panics if the handle is not
    /// associated with this `ObjectStore`. This is very cheap: about as cheap as a bounds-checked
    /// index into an array.
    pub fn get_mut<T>(&mut self, handle: Handle<T>) -> &mut T
    where
        T: ?Sized + 'static,
    {
        self.inner.get_mut(handle)
    }

    /// Returns the [`TypeId`] of the concrete type stored at the given handle. The returned type
    /// id will not match `T` if `T` is a trait object type. Panics if the input handle is not
    /// associated with this `ObjectStore`.
    pub fn get_type_id<T>(&self, handle: Handle<T>) -> TypeId
    where
        T: ?Sized,
    {
        self.inner.get_type_id(handle)
    }

    /// Returns the [`TypeId`] of the concrete type stored at the given handle. Panics if the input
    /// handle is not associated with this `ObjectStore`.
    pub fn get_type_id_dynamic(&self, handle: DynamicHandle) -> TypeId {
        self.inner.get_type_id_dynamic(handle)
    }

    /// Gets the name of the type stored at the given handle. Panics if the input handle is not
    /// associated with this `ObjectStore`.
    pub fn get_type_name<T>(&self, handle: Handle<T>) -> &str
    where
        T: ?Sized,
    {
        self.inner.get_type_name(handle)
    }

    /// Gets the name of the type stored at the given dynamic handle. Panics if the input handle is
    /// not associated with this `ObjectStore`.
    pub fn get_type_name_dynamic(&self, handle: DynamicHandle) -> &str {
        self.inner.get_type_name_dynamic(handle)
    }
}

/// The object store as `Send` if its tag is `Send` because this ensures that all contained objects
/// are `Send`.
unsafe impl<Tag> Send for ObjectStore<Tag> where Tag: Send {}

/// The object store as `Sync` if its tag is `Sync` because this ensures that all contained objects
/// are `Sync`.
unsafe impl<Tag> Sync for ObjectStore<Tag> where Tag: Sync {}

// ------------------------------------------------------------------------------

impl UninitObjectStore<tag::ThreadLocal> {
    /// Instantiates a new [`UninitObjectStore`] object that can store objects which are `!Send` and
    /// `!Sync`.
    pub fn new_thread_local() -> Self {
        Self::new()
    }
}

impl UninitObjectStore<tag::Send> {
    /// Instantiates a new [`UninitObjectStore`] object that can store objects which are `Send` but
    /// `!Sync`.
    pub fn new_send() -> Self {
        Self::new()
    }
}

impl UninitObjectStore<tag::SendSync> {
    /// Instantiates a new [`UninitObjectStore`] object that can store objects which are `Send` and `Sync`.
    pub fn new_sync() -> Self {
        Self::new()
    }
}

impl<Tag> UninitObjectStore<Tag> {
    /// Instantiates a new [`UninitObjectStore`] object.
    pub fn new() -> Self {
        UninitObjectStore {
            _marker: std::marker::PhantomData,
            inner: ObjectStoreInner::new(),
            unwritten_objects: Default::default(),
        }
    }

    /// If every reserved handle has been written to, returns [`Ok`] with an initialized
    /// [`ObjectStore`].  Otherwise, returns `Err(self)`.
    pub fn try_init(self) -> Result<ObjectStore<Tag>, UninitObjectStore<Tag>> {
        if self.unwritten_objects.is_empty() {
            Ok(ObjectStore {
                _marker: self._marker,
                inner: self.inner,
            })
        } else {
            Err(self)
        }
    }

    /// Iterates over unwritten handles in this store. Intended mainly for diagnostics.
    pub fn iter_unwritten_handles<'a>(
        &'a self,
    ) -> impl Iterator<Item = (DynamicHandle, UninitObjectInfo)> + 'a {
        self.unwritten_objects.iter().map(|(&ptr, &info)| {
            (
                DynamicHandle {
                    inner: Handle {
                        store_id: self.inner.id,
                        meta_ix: info.meta_ix,
                        ptr,
                    },
                },
                info,
            )
        })
    }

    /// Returns the number of objects in this store that are yet to be written.
    pub fn count_unwritten_handles(&self) -> usize {
        self.unwritten_objects.len()
    }

    /// Returns true if a call to [`UninitObjectStore::try_init`] will succeed.
    pub fn can_init(&self) -> bool {
        self.count_unwritten_handles() == 0
    }

    /// Pushes a reservation for a new object to the given buffer and returns a handle for it. Any
    /// [`u32`] can be chosen as the `buffer_ix` but bear in mind that a large choice of index will
    /// cause a large allocation to happen if smaller buffer indexes have not yet been used. This
    /// handle must later be written to with [`UninitObjectStore::write`], otherwise initializing
    /// the store will panic.
    pub fn reserve<T>(&mut self, buffer_ix: u32) -> Handle<T>
    where
        T: Castable + tag::Satisfies<Tag>,
    {
        let handle = self.inner.reserve(buffer_ix);
        self.unwritten_objects.insert(
            handle.ptr as *const (),
            UninitObjectInfo {
                buffer_ix,
                meta_ix: handle.meta_ix,
            },
        );
        handle
    }

    /// If the given handle has not already been written to, writes the given value to it.
    /// Otherwise, panics. Also panics if the handle is not associated with this object store.
    pub fn write<T>(&mut self, handle: Handle<T>, value: T)
    where
        T: Sized,
    {
        if !self.try_write(handle, value) {
            panic!("writing to the same handle twice");
        }
    }

    /// If the given handle has not already been written to, writes the given value to it and
    /// returns `true`. Otherwise, returns `false`. Panics if the handle is not associated with
    /// this object store.
    pub fn try_write<T>(&mut self, handle: Handle<T>, value: T) -> bool
    where
        T: Sized,
    {
        self.unwritten_objects
            .remove(&(handle.ptr as *const ()))
            .and_then(|_| {
                // SAFETY:
                // a) the T: Sized bound ensures that T is in fact the concrete type of the data,
                // or at least a type trivially transmutable to it
                // b) the unwritten_objects check ensures that we can't write to the same handle
                // twice
                unsafe {
                    self.inner.write(handle, value);
                }
                Some(())
            })
            .is_some()
    }

    /// Reserves a new object in the given buffer and returns a handle for it. Any [`u32`] can be
    /// chosen as the `buffer_ix` but bear in mind that a large choice of index will cause a large
    /// allocation to happen if smaller buffer indexes have not yet been used.
    pub fn push<T>(&mut self, buffer_ix: u32, item: T) -> Handle<T>
    where
        T: Castable + tag::Satisfies<Tag>,
    {
        self.inner.push(buffer_ix, item)
    }

    /// Iterates over handles for all stored objects in the given buffer that are castable to the
    /// requested type. This is a fast operation implemented as a range lookup in a BTreeSet.
    pub fn find<'a, T>(&'a self, buffer_ix: u32) -> impl Iterator<Item = Handle<T>> + 'a
    where
        T: ?Sized + 'static,
    {
        self.inner.find::<T>(buffer_ix)
    }

    /// Iterates over dynamic handles for all stored objects in the given buffer that are castable
    /// to the requested type. This is a fast operation implemented as a range lookup in a
    /// BTreeSet.
    pub fn find_dynamic<'a>(
        &'a self,
        buffer_ix: u32,
        type_id: TypeId,
    ) -> impl Iterator<Item = DynamicHandle> + 'a {
        self.inner.find_dynamic(buffer_ix, type_id)
    }

    /// Casts an untyped `DynamicHandle` up to a typed `Handle`. Panics if the input handle is not
    /// associated with this `UninitObjectStore`. Returns `None` if the object that the handle
    /// points to is not castable to the requested type.
    pub fn cast_from_dynamic<T>(&self, handle: DynamicHandle) -> Option<Handle<T>>
    where
        T: ?Sized + 'static,
    {
        self.inner.cast_from_dynamic(handle)
    }

    /// Casts a handle down to an untyped DynamicHandle. Panics if the input handle is not
    /// associated with this `UninitObjectStore`.
    pub fn cast_to_dynamic<T>(&self, handle: Handle<T>) -> DynamicHandle
    where
        T: ?Sized,
    {
        self.inner.cast_to_dynamic(handle)
    }

    /// Casts a handle to a different type. Panics if the input handle is not associated with this
    /// `UninitObjectStore`. Returns `None` if the concrete type pointed to by the input handle is
    /// not castable to `U`.
    pub fn cast<T, U>(&self, handle: Handle<T>) -> Option<Handle<U>>
    where
        T: ?Sized + 'static,
        U: ?Sized + 'static,
    {
        self.inner.cast(handle)
    }

    /// Returns the [`TypeId`] of the concrete type stored at the given handle. The returned type
    /// id will not match `T` if `T` is a trait object type. Panics if the input handle is not
    /// associated with this `UninitObjectStore`.
    pub fn get_type_id<T>(&self, handle: Handle<T>) -> TypeId
    where
        T: ?Sized,
    {
        self.inner.get_type_id(handle)
    }

    /// Returns the [`TypeId`] of the concrete type stored at the given handle. Panics if the input
    /// handle is not associated with this `UninitObjectStore`.
    pub fn get_type_id_dynamic(&self, handle: DynamicHandle) -> TypeId {
        self.inner.get_type_id_dynamic(handle)
    }

    /// Gets the name of the type stored at the given handle. Panics if the input handle is not
    /// associated with this `UninitObjectStore`.
    pub fn get_type_name<T>(&self, handle: Handle<T>) -> &str
    where
        T: ?Sized,
    {
        self.inner.get_type_name(handle)
    }

    /// Gets the name of the type stored at the given dynamic handle. Panics if the input handle is
    /// not associated with this `UninitObjectStore`.
    pub fn get_type_name_dynamic(&self, handle: DynamicHandle) -> &str {
        self.inner.get_type_name_dynamic(handle)
    }
}

/// An uninit object store as `Send` as long as its tag is `Send` because the tag ensures that all
/// stored objects are `Send`.
unsafe impl<Tag> Send for UninitObjectStore<Tag> where Tag: Send {}

/// An uninit object store can be safely accessed from any thread regardless of its tag because
/// objects inside it cannot be accessed while it is uninitialized.
unsafe impl<Tag> Sync for UninitObjectStore<Tag> {}

// ------------------------------------------------------------------------------

impl ObjectMeta {
    fn new<T>() -> Self
    where
        T: Castable,
    {
        Self {
            name: T::name().to_string(),
            type_id: TypeId::of::<T>(),
            // SAFETY: destroy() is only called in this module when the target is known to be an
            // instance of T and where the destructor will not be run another time
            destroy: |data: *mut ()| unsafe { destroy::<T>(data) },
        }
    }
}

// ------------------------------------------------------------------------------

// SAFETY: access only happens through the ObjectStore which is only Send or Sync if all objects
// stored inside it are. Inspecting a handle on another thread is always safe since it is always
// impossible to access data through an ObjectStore that did not handle out the handle in the first
// place.
unsafe impl<T: ?Sized + 'static> Sync for Handle<T> {}
unsafe impl<T: ?Sized + 'static> Send for Handle<T> {}

impl<T> Clone for Handle<T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        Handle {
            store_id: self.store_id,
            meta_ix: self.meta_ix,
            ptr: self.ptr,
        }
    }
}

impl<T> Copy for Handle<T> where T: ?Sized {}

impl<T> PartialEq for Handle<T>
where
    T: ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> Eq for Handle<T> where T: ?Sized {}

impl<T> PartialOrd for Handle<T>
where
    T: ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ptr.partial_cmp(&other.ptr)
    }
}

impl<T> Ord for Handle<T>
where
    T: ?Sized,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ptr.cmp(&other.ptr)
    }
}

impl<T> std::hash::Hash for Handle<T>
where
    T: ?Sized,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.ptr.hash(hasher)
    }
}

impl<T> std::fmt::Debug for Handle<T>
where
    T: ?Sized + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("Handle")
            .field("type", &TypeId::of::<T>())
            .field("store_id", &self.store_id)
            .field("meta_ix", &self.meta_ix)
            .field("ptr", &self.ptr)
            .finish()
    }
}

// ------------------------------------------------------------------------------

fn new_store_id() -> u32 {
    static NEXT_ID: once_cell::sync::OnceCell<Mutex<u32>> = once_cell::sync::OnceCell::new();
    let mut next_id = NEXT_ID.get_or_init(|| Mutex::new(0)).lock().unwrap();

    if *next_id == std::u32::MAX {
        panic!(
            "total number of [`ObjectStore`] instances over the life of the program reached {} so
            no more stores can be created",
            *next_id
        );
    }

    let id = *next_id;
    *next_id += 1;
    id
}

impl ObjectStoreInner {
    fn new() -> Self {
        ObjectStoreInner {
            id: new_store_id(),
            casts: Default::default(),
            metas: Default::default(),
            metas_by_type: Default::default(),
            metas_by_cast_target: Default::default(),
            object_keys: Default::default(),
            buffers: Default::default(),
        }
    }

    fn reserve<T>(&mut self, buffer_ix: u32) -> Handle<T>
    where
        T: Castable,
    {
        let meta_ix = *self
            .metas_by_type
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                let meta = ObjectMeta::new::<T>();
                let ix = self.metas.len() as u32;
                self.metas.push(meta);
                T::collect_casts(&mut self.casts);

                for (dst, _) in self.casts.find_keys_by_src(TypeId::of::<T>()) {
                    self.metas_by_cast_target.insert((dst, ix));
                }

                ix
            });

        while self.buffers.len() <= buffer_ix as usize {
            self.buffers.push(Default::default());
        }

        let buffer = &mut self.buffers[buffer_ix as usize];
        let (ptr, offset) = buffer.reserve::<T>();

        self.object_keys.insert(ObjectKey {
            buffer_ix,
            meta_ix,
            offset,
        });

        Handle {
            store_id: self.id,
            meta_ix,
            ptr,
        }
    }

    // SAFETY:
    // a) T must be the concrete type of the provided handle
    // b) the handle must not have been written to already
    unsafe fn write<T: Sized>(&mut self, handle: Handle<T>, item: T) {
        let addr = handle.ptr as *mut T;
        std::ptr::write(addr, item);
    }

    fn push<T>(&mut self, buffer_ix: u32, item: T) -> Handle<T>
    where
        T: Castable,
    {
        let handle = self.reserve::<T>(buffer_ix);

        // SAFETY: reserve() guarantees that the returned pointer is a suitable memory location to
        // write an object of type T and that no references to that memory exist yet.
        unsafe {
            self.write(handle, item);
        }

        handle
    }

    fn find<'a, T>(&'a self, buffer_ix: u32) -> impl Iterator<Item = Handle<T>> + 'a
    where
        T: ?Sized + 'static,
    {
        self.find_dynamic(buffer_ix, TypeId::of::<T>())
            .flat_map(|handle| self.cast_from_dynamic::<T>(handle))
    }

    fn find_dynamic<'a>(
        &'a self,
        buffer_ix: u32,
        type_id: TypeId,
    ) -> impl Iterator<Item = DynamicHandle> + 'a {
        self.metas_by_cast_target
            .range((type_id, 0)..=(type_id, std::u32::MAX))
            .flat_map(move |&(_, meta_ix)| {
                self.object_keys
                    .range(
                        ObjectKey {
                            buffer_ix,
                            meta_ix,
                            offset: (0, 0),
                        }..ObjectKey {
                            buffer_ix,
                            meta_ix: 1 + meta_ix,
                            offset: (0, 0),
                        },
                    )
                    .map(move |&key| self.key_to_handle(key))
            })
    }

    fn key_to_handle(&self, key: ObjectKey) -> DynamicHandle {
        DynamicHandle {
            inner: Handle {
                store_id: self.id,
                meta_ix: key.meta_ix,
                ptr: self.buffers[key.buffer_ix as usize].offset(key.offset),
            },
        }
    }

    fn cast_from_dynamic<T>(&self, handle: DynamicHandle) -> Option<Handle<T>>
    where
        T: ?Sized + 'static,
    {
        let meta = &self.metas[handle.inner.meta_ix as usize];
        let cast_ix = self.casts.find_key(meta.type_id, TypeId::of::<T>())?;

        Some(Handle {
            store_id: handle.inner.store_id,
            meta_ix: handle.inner.meta_ix,
            ptr: self.casts.cast(cast_ix, handle.inner.ptr),
        })
    }

    fn cast_to_dynamic<T>(&self, handle: Handle<T>) -> DynamicHandle
    where
        T: ?Sized,
    {
        assert_eq!(handle.store_id, self.id);
        DynamicHandle {
            inner: Handle {
                meta_ix: handle.meta_ix,
                store_id: self.id,
                ptr: handle.ptr as *const (),
            },
        }
    }

    fn cast<T, U>(&self, handle: Handle<T>) -> Option<Handle<U>>
    where
        T: ?Sized + 'static,
        U: ?Sized + 'static,
    {
        self.cast_from_dynamic(self.cast_to_dynamic(handle))
    }

    fn get<T>(&self, handle: Handle<T>) -> &T
    where
        T: ?Sized + 'static,
    {
        assert_eq!(handle.store_id, self.id);

        // SAFETY: Since we have checked that the handle's store id matches our id, the handle can
        // only have been acquired from this store instance. Objects within the store are never
        // removed or moved around in buffers, so reading this type from this handle's pointer is
        // guaranteed to be valid.
        unsafe { &*handle.ptr }
    }

    fn get_mut<T>(&mut self, handle: Handle<T>) -> &mut T
    where
        T: ?Sized + 'static,
    {
        assert_eq!(handle.store_id, self.id);

        // SAFETY: Since we have checked that the handle's store id matches our id, the handle can
        // only have been acquired from this store instance. Objects within the store are never
        // removed or moved around in buffers, so reading this type from this handle's pointer is
        // guaranteed to be valid.
        //
        // It is safe to form the mutable reference because we have a mutable reference to `self`
        // guaranteeing that nothing else can be referencing anything inside `self`.
        unsafe { &mut *(handle.ptr as *mut T) }
    }

    fn get_type_id<T>(&self, handle: Handle<T>) -> TypeId
    where
        T: ?Sized,
    {
        assert_eq!(handle.store_id, self.id);
        self.metas[handle.meta_ix as usize].type_id
    }

    fn get_type_id_dynamic(&self, handle: DynamicHandle) -> TypeId {
        assert_eq!(handle.inner.store_id, self.id);
        self.metas[handle.inner.meta_ix as usize].type_id
    }

    fn get_type_name<T>(&self, handle: Handle<T>) -> &str
    where
        T: ?Sized,
    {
        assert_eq!(handle.store_id, self.id);
        self.metas[handle.meta_ix as usize].name.as_str()
    }

    fn get_type_name_dynamic(&self, handle: DynamicHandle) -> &str {
        assert_eq!(handle.inner.store_id, self.id);
        self.metas[handle.inner.meta_ix as usize].name.as_str()
    }
}

impl Drop for ObjectStoreInner {
    fn drop(&mut self) {
        for key in &self.object_keys {
            let ptr = self.buffers[key.buffer_ix as usize].offset_mut(key.offset);
            let meta = &self.metas[key.meta_ix as usize];
            (meta.destroy)(ptr);
        }
    }
}

// ------------------------------------------------------------------------------

/// SAFETY: it is only safe to call this when the object that `ptr` points to is an instance of `T`
/// and it will not be dropped again.
unsafe fn destroy<T>(ptr: *mut ())
where
    T: Sized,
{
    let ptr = ptr as *mut T;
    std::ptr::drop_in_place(ptr);
}

// ------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use super::{Handle, ObjectStore, UninitObjectStore};

    trait Object {}

    #[derive(PartialEq, Eq, Debug)]
    struct MyObjectA {
        a: String,
        b: i32,
    }

    #[derive(PartialEq, Eq, Debug)]
    struct MyObjectB {
        x: u32,
    }

    crate::impl_castable! {
        impl Castable for MyObjectA {
            into dyn std::fmt::Debug;
            into dyn std::any::Any;
            into dyn Object;
        }
    }

    crate::impl_castable! {
        impl Castable for MyObjectB {
            into dyn Object;
        }
    }

    impl Object for MyObjectA {}
    impl Object for MyObjectB {}

    #[test]
    fn test_object_store() {
        let mut store = ObjectStore::new_send();

        let handle0 = store.push(
            0,
            MyObjectA {
                a: "test".to_string(),
                b: 4,
            },
        );
        let handle1 = store.push(0, MyObjectB { x: 0 });
        let handle2 = store.push(
            2,
            MyObjectA {
                a: "bar".to_string(),
                b: 8,
            },
        );
        let handle3 = store.push(
            0,
            MyObjectA {
                a: "baz".to_string(),
                b: 1,
            },
        );

        check_store_contents(
            &store,
            TestHandles {
                handle0,
                handle1,
                handle2,
                handle3,
            },
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_object_store() {
        let mut store1 = ObjectStore::new_send();
        let store2 = ObjectStore::new_send();

        let handle = store1.push(
            0,
            MyObjectA {
                a: "test".to_string(),
                b: 4,
            },
        );

        // access through a handle that was assigned from a different store panics in order to
        // prevent invalid memory acess
        store2.get(handle);
    }

    #[test]
    fn test_uninit_object_store() {
        let mut store = UninitObjectStore::new_thread_local();

        let handle0 = store.reserve::<MyObjectA>(0);
        let handle1 = store.push(0, MyObjectB { x: 0 });
        let handle2 = store.reserve::<MyObjectA>(2);
        let handle3 = store.reserve::<MyObjectA>(0);

        assert_eq!(store.count_unwritten_handles(), 3);
        let mut store = match store.try_init() {
            Ok(_) => panic!("store should not yet be initializable"),
            Err(store) => store,
        };

        store.write(
            handle0,
            MyObjectA {
                a: "test".to_string(),
                b: 4,
            },
        );
        assert_eq!(store.count_unwritten_handles(), 2);

        // cannot write to the same handle twice
        assert!(!store.try_write(
            handle0,
            MyObjectA {
                a: "test".to_string(),
                b: 4,
            },
        ));
        assert!(!store.try_write(handle1, MyObjectB { x: 0 }));
        assert_eq!(store.count_unwritten_handles(), 2);

        store.write(
            handle2,
            MyObjectA {
                a: "bar".to_string(),
                b: 8,
            },
        );
        assert!(store.try_write(
            handle3,
            MyObjectA {
                a: "baz".to_string(),
                b: 1,
            },
        ));
        assert_eq!(store.count_unwritten_handles(), 0);
        assert!(store.can_init());

        let store = match store.try_init() {
            Ok(store) => store,
            Err(_) => {
                panic!("store should be initializable after all handles have been written to")
            }
        };

        check_store_contents(
            &store,
            TestHandles {
                handle0,
                handle1,
                handle2,
                handle3,
            },
        );
    }

    struct TestHandles {
        handle0: Handle<MyObjectA>,
        handle1: Handle<MyObjectB>,
        handle2: Handle<MyObjectA>,
        handle3: Handle<MyObjectA>,
    }

    fn check_store_contents<Tag>(store: &ObjectStore<Tag>, handles: TestHandles) {
        let TestHandles {
            handle0,
            handle1,
            handle2,
            handle3,
        } = handles;

        // `get` allows access through handles
        assert_eq!(
            store.get(handle0),
            &MyObjectA {
                a: "test".to_string(),
                b: 4
            }
        );
        assert_eq!(store.get(handle1), &MyObjectB { x: 0 });
        assert_eq!(
            store.get(handle2),
            &MyObjectA {
                a: "bar".to_string(),
                b: 8
            }
        );
        assert_eq!(
            store.get(handle3),
            &MyObjectA {
                a: "baz".to_string(),
                b: 1
            }
        );

        // `find_dynamic` returns the right handles
        assert_eq!(
            store
                .find_dynamic(0, TypeId::of::<MyObjectA>())
                .next()
                .unwrap(),
            store.cast_to_dynamic(handle0)
        );
        assert_eq!(
            store
                .find_dynamic(2, TypeId::of::<MyObjectA>())
                .next()
                .unwrap(),
            store.cast_to_dynamic(handle2)
        );

        // `find` returns all objects with matching types
        {
            let objects: Vec<_> = store.find::<MyObjectA>(0).collect();
            assert_eq!(objects, vec![handle0, handle3]);
        }
        {
            let objects: Vec<_> = store.find::<MyObjectA>(2).collect();
            assert_eq!(objects, vec![handle2]);
        }
        {
            let objects: Vec<_> = store.find::<MyObjectB>(0).collect();
            assert_eq!(objects, vec![handle1]);
        }
        {
            let objects: Vec<_> = store.find::<MyObjectB>(2).collect();
            assert_eq!(objects, vec![]);
        }

        {
            // `find` also works for types castable from the concrete object types and the found
            // handles can be used to access a reference to the target trait object
            let debug_handles: Vec<_> = store.find::<dyn std::fmt::Debug>(0).collect();
            assert_eq!(debug_handles.len(), 2);
            assert_eq!(
                format!("{:?}", store.get(debug_handles[0])),
                "MyObjectA { a: \"test\", b: 4 }"
            );
            assert_eq!(
                format!("{:?}", store.get(debug_handles[1])),
                "MyObjectA { a: \"baz\", b: 1 }"
            );
        }

        {
            // `find` is able to return objects with heterogeneous concrete types
            let handles: Vec<_> = store.find::<dyn Object>(0).collect();
            assert_eq!(handles.len(), 3);
            assert_eq!(
                store.cast_to_dynamic(handles[0]),
                store.cast_to_dynamic(handle0)
            );
            assert_eq!(
                store.cast_to_dynamic(handles[1]),
                store.cast_to_dynamic(handle3)
            );
            assert_eq!(
                store.cast_to_dynamic(handles[2]),
                store.cast_to_dynamic(handle1)
            );
        }
    }
}
