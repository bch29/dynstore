use std::any::TypeId;
use std::collections::BTreeMap;

/// Types implementing this trait know how to cast themselves into trait objects for some of the
/// traits that they implement.
///
/// This trait should only be implemented via the [`impl_castable!`] macro.
pub trait Castable: 'static {
    /// Returns the name of this type.
    fn name() -> &'static str;

    /// Populates the given [`Casts`] object with all of the known ways to cast out of this type.
    fn collect_casts(casts: &mut Casts)
    where
        Self: Sized;
}

/// Stores known ways to cast from concrete types to trait objects.
#[derive(Default)]
pub struct Casts {
    // (src, dst, ptr)
    casts: Vec<(TypeId, TypeId, *const ())>,

    // options allow us to search for a range without having something to put in the second key
    // field
    casts_by_dst_src: BTreeMap<(Option<TypeId>, Option<TypeId>), u32>,
    casts_by_src_dst: BTreeMap<(Option<TypeId>, Option<TypeId>), u32>,
}

impl Casts {
    /// Adds a cast from T to U.
    ///
    /// ## Safety ##
    /// The provided function must not do anything except trivially return the pointer that it is
    /// given, potentially with a vtable attached.
    pub unsafe fn add<T, U>(&mut self, cast: fn(*const T) -> *const U)
    where
        T: 'static,
        U: ?Sized + 'static,
    {
        let src = TypeId::of::<T>();
        let dst = TypeId::of::<U>();
        self.casts_by_src_dst
            .insert((Some(src), Some(dst)), self.casts.len() as u32);
        self.casts_by_dst_src
            .insert((Some(dst), Some(src)), self.casts.len() as u32);

        self.casts.push((src, dst, cast as *const ()));
    }

    /// Looks up an index that can be used to quickly access a cast from the source type to the
    /// destination type, via [`Casts::cast`].
    pub fn find_key(&self, src_type: TypeId, dst_type: TypeId) -> Option<u32> {
        self.casts_by_dst_src
            .get(&(Some(dst_type), Some(src_type)))
            .copied()
    }

    /// Casts the input type-erased pointer into the destination type. The resulting pointer is
    /// only safe to dereference if the underlying type behind the src pointer is the right source
    /// type for the given cast index.
    pub fn cast<Dst>(&self, key: u32, src: *const ()) -> *const Dst
    where
        Dst: ?Sized + 'static,
    {
        let (_t0, t1, ptr) = dbg!(self.casts[key as usize]);
        assert_eq!(t1, TypeId::of::<Dst>());

        // SAFETY: this pointer came from `.add()` which provides a function pointer with the same
        // type except that originally the input was not type-erased.
        let func: fn(*const ()) -> *const Dst = unsafe { std::mem::transmute(ptr) };
        func(src)
    }

    /// Looks up the destination type for the given cast.
    pub fn get_dst(&self, key: u32) -> TypeId {
        self.casts[key as usize].0
    }

    /// Looks up the source type for the given cast.
    pub fn get_src(&self, key: u32) -> TypeId {
        self.casts[key as usize].1
    }

    /// Iterates over known casts into the given target type. Items in the returned iterator are
    /// (source type, cast key).
    pub fn find_keys_by_dst<'a>(&'a self, ty: TypeId) -> impl Iterator<Item = (TypeId, u32)> + 'a {
        self.casts_by_dst_src
            .range((Some(ty), None)..)
            .take_while(move |((dst, _src), _)| dst == &Some(ty))
            .map(|((_dst, src), ix)| (src.unwrap(), *ix))
    }

    /// Iterates over known casts out of the given source type. Items in the returned iterator are
    /// (target type, cast key).
    pub fn find_keys_by_src<'a>(&'a self, ty: TypeId) -> impl Iterator<Item = (TypeId, u32)> + 'a {
        self.casts_by_src_dst
            .range((Some(ty), None)..)
            .take_while(move |((src, _dst), _)| src == &Some(ty))
            .map(|((_src, dst), ix)| (dst.unwrap(), *ix))
    }
}

// SAFETY: send and sync are safe because the raw pointers inside `Casts` are only static function
// pointers
unsafe impl Send for Casts {}
unsafe impl Sync for Casts {}

#[doc(hidden)]
#[macro_export]
macro_rules! add_cast {
    ($casts:expr, $src:ty, $dst:ty) => {
        unsafe {
            $casts.add::<$src, $dst>(|x| x);
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_castable_from {
    {
        impl Castable for $struct:ident {
            $(
                into dyn $trait:path;
            )*
        }
    } => {
        impl $crate::Castable for $struct {
            fn name() -> &'static str {
                stringify!($struct)
            }

            fn collect_casts(casts: &mut $crate::Casts) {
                $crate::add_cast!(casts, $struct, $struct);
                $(
                    $crate::add_cast!(casts, $struct, dyn $trait);
                )*

                #[cfg(feature = "inventory")]
                for register in inventory::iter::<$crate::RegisterCast> {
                    if (register.cast_from)() == std::any::TypeId::of::<$struct>() {
                        (register.register)(casts)
                    }
                }
            }
        }
    };
}

#[doc(hidden)]
pub const fn assert_implements_castable<T>()
where
    T: Castable,
{
}

#[cfg(feature = "inventory")]
#[macro_export]
#[doc(hidden)]
macro_rules! impl_castable_into {
    {
        impl Castable into dyn $trait:path {
            $(for $struct:path;)*
        }
    } => {
        $(
            const _: () = $crate::assert_implements_castable::<$struct>();
            inventory::submit! {
                $crate::RegisterCast {
                    cast_from: || std::any::TypeId::of::<$struct>(),
                    register: |casts| $crate::add_cast!(casts, $struct, dyn $trait),
                }
            }
        )*
    }
}


#[cfg(feature = "inventory")]
/// Implements [`Castable`] for a struct.
///
/// # Examples
/// ```
/// #[derive(Debug)]
/// struct Foo;
///
/// trait Baz {}
/// impl Baz for Foo {}
///
/// dynstore::impl_castable! {
///     impl Castable for Foo {
///         into dyn std::fmt::Debug;
///         into dyn Baz;
///     }
/// }
/// ```
///
/// If the `inventory` feature is enabled, you can use another variation of this macro to enable
/// casting into a new trait for an already-[`Castable`] struct.
/// ```
/// pub struct Foo;
/// pub struct Bar;
/// pub trait Baz {}
/// impl Baz for Foo {}
/// dynstore::impl_castable! {
///     impl Castable for Foo {
///         into dyn Baz;
///     }
///     impl Castable for Bar {}
/// }
///
/// trait Bing {}
/// impl Bing for Foo {}
/// impl Bing for Bar {}
/// dynstore::impl_castable! {
///     impl Castable into dyn Bing {
///         for Foo;
///         for Bar;
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_castable {
    {} => {};
    {
        impl Castable for $struct:ident {
            $(
                into dyn $trait:path;
            )*
        } $($tail:tt)*
    } => {
        $crate::impl_castable_from! {
            impl Castable for $struct {
                $(
                    into dyn $trait;
                 )*
            }
        }
        $crate::impl_castable!{$($tail)*}
    };
    {
        impl Castable into dyn $trait:path {
            $(for $struct:path;)*
        } $($tail:tt)*
    } => {
        $crate::impl_castable_into! {
            impl Castable into dyn $trait {
                $(for $struct;)*
            }
        }
        $crate::impl_castable!{$($tail)*}
    }
}

#[cfg(not(feature = "inventory"))]
/// Implements [`Castable`] for a struct.
///
/// # Example
/// ```
/// #[derive(Debug)]
/// struct Foo;
///
/// trait MyTrait {}
/// impl MyTrait for Foo {}
///
/// dynstore::impl_castable! {
///     impl Castable for Foo {
///         into dyn std::fmt::Debug;
///         into dyn MyTrait;
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_castable {
    {
        $(impl Castable for $struct:ident {
            $(
                into dyn $trait:path;
            )*
        })*
    } => {
        $($crate::impl_castable_from! {
            impl Castable for $struct {
                $(
                    into dyn $trait;
                 )*
            }
        })*
    }
}

#[doc(hidden)]
#[cfg(feature = "inventory")]
pub struct RegisterCast {
    pub cast_from: fn() -> TypeId,
    pub register: fn(&mut Casts),
}

#[cfg(feature = "inventory")]
inventory::collect!(RegisterCast);

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::collections::BTreeSet;

    use super::Castable;

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
    fn test_casts() {
        assert_eq!(MyObjectA::name(), "MyObjectA");

        let mut casts = Default::default();
        MyObjectA::collect_casts(&mut casts);
        MyObjectB::collect_casts(&mut casts);

        assert_eq!(
            casts.find_key(TypeId::of::<MyObjectA>(), TypeId::of::<MyObjectA>()),
            Some(0)
        );
        assert_eq!(
            casts.find_key(TypeId::of::<MyObjectA>(), TypeId::of::<dyn Object>()),
            Some(3)
        );
        assert_eq!(
            casts.find_key(TypeId::of::<MyObjectB>(), TypeId::of::<dyn Object>()),
            Some(5)
        );

        assert_eq!(
            casts
                .find_keys_by_dst(TypeId::of::<dyn Object>())
                .collect::<BTreeSet<(TypeId, u32)>>(),
            vec![
                (TypeId::of::<MyObjectA>(), 3),
                (TypeId::of::<MyObjectB>(), 5)
            ]
            .into_iter()
            .collect()
        );

        assert_eq!(
            casts
                .find_keys_by_src(TypeId::of::<MyObjectA>())
                .collect::<BTreeSet<(TypeId, u32)>>(),
            vec![
                (TypeId::of::<MyObjectA>(), 0),
                (TypeId::of::<dyn std::fmt::Debug>(), 1),
                (TypeId::of::<dyn std::any::Any>(), 2),
                (TypeId::of::<dyn Object>(), 3)
            ]
            .into_iter()
            .collect()
        );

        assert_eq!(
            casts
                .find_keys_by_src(TypeId::of::<MyObjectB>())
                .collect::<BTreeSet<(TypeId, u32)>>(),
            vec![
                (TypeId::of::<MyObjectB>(), 4),
                (TypeId::of::<dyn Object>(), 5)
            ]
            .into_iter()
            .collect()
        );
    }
}
