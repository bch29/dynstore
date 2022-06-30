Provides the `ObjectStore` type, a container which can store arbitrary objects as long as
they implement `Castable`.

Provides lifetime-free handles which can later be safely used to access
references to the stored objects.  Supports casting handles to trait objects for any trait
implemented by a stored object and registered in its `Castable` instance.  The internal data
layout efficiently packs objects, storing minimal metadata for type casting and searches
separately.

# Variants

Supports variants dependending on the kind of types you need to store:
- `Send` and `Sync`: `ObjectStore<tag::SendSync>`
- `Send` but not `Sync`: `ObjectStore<tag::Send>`
- neither `Send` nor `Sync`: `ObjectStore<tag::ThreadLocal>`

# Buffers

The `ObjectStore` contains multiple conceptual "buffers" keyed by a `u32`. When you push an
object to the store, you have to choose which buffer index it gets assigned to. In the simplest
usage, only one buffer (index 0) can be used. However, making use of multiple buffers can have
performance benefits:
- Objects pushed to the same buffer are stored near to each other in memory.
- `ObjectStore::find` operates on a single buffer so the buffer index can be treated as a
primary key to speed up searching.

# Caveats

The motivating use case is to store objects that live for the entire lifetime of the program.
Thus, once objects are pushed to the `ObjectStore`, they are not dropped until the whole
store is dropped. Due to the way handles are implemented, there is also a limitation that at
most 2^32 `ObjectStore` instances can be created over the whole life of a program. This might
limit some use cases as a temporary arena allocator.

Pushing objects into the store is not fast enough to do in a hot loop. It involves a hash
lookup and a few indirections in the common case, and some allocations and linear searches
whenever an object of a previously-unseen type is pushed. The store is primarily designed for
fast access, searches and casts, not fast allocation.

Internally, memory is segmented into allocations of size 2MB. This has three main consequences
that users need to be aware of:
- Any attempt to push an individual object larger than 2MB to an [`ObjectStore`] will panic.
- If object sizes are a significant fraction of 2MB, memory can be wasted at the end of each
segment.
- At least 2MB will be allocated per buffer index used. If 1000 different buffer indexes are
used, this is 2GB which is quite significant. It is therefore not efficient to store a few
small objects across many buffers.
