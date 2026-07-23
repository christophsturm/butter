// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-ffi
//!
//! C-ABI bridge over the shared FFAI engine. This is how **Swift consumes
//! the same Rust layer the CUDA backend uses**: build this crate as a
//! static/dynamic library, link it from Swift (or any C caller), and drive
//! the [`ffai_core::Device`] layer through these `extern "C"` entry points.
//!
//! On Apple hardware the Swift-native Metal engine remains the primary
//! path, so [`ffai_open_device`] returns null there until the Rust Metal
//! backend lands — but the bridge itself (link + call into the shared
//! engine) is proven, and on a CUDA host the same calls drive the live GB10
//! through `ffai-cuda`.

use ffai::Device;
use std::ffi::{CString, c_char};
use std::sync::Arc;

/// Heap-allocate a C string for return. Free with [`ffai_string_free`].
fn cstr(s: String) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

/// Engine version (the `ffai` crate version).
#[unsafe(no_mangle)]
pub extern "C" fn ffai_version() -> *mut c_char {
    cstr(env!("CARGO_PKG_VERSION").to_string())
}

/// Comma-separated list of backends compiled into this build.
#[unsafe(no_mangle)]
pub extern "C" fn ffai_compiled_backends() -> *mut c_char {
    cstr(ffai::compiled_backends().join(","))
}

/// Free a string returned by any `ffai_*` function.
#[unsafe(no_mangle)]
pub extern "C" fn ffai_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Opaque handle to a live [`Device`] held across the FFI boundary.
pub struct FfaiDevice(Arc<dyn Device>);

/// Open the first live device, or null if none is available on this host.
#[unsafe(no_mangle)]
pub extern "C" fn ffai_open_device() -> *mut FfaiDevice {
    match ffai::devices().into_iter().next() {
        Some(d) => Box::into_raw(Box::new(FfaiDevice(d))),
        None => std::ptr::null_mut(),
    }
}

/// Backend name of a device handle (e.g. `"cuda"`).
#[unsafe(no_mangle)]
pub extern "C" fn ffai_device_backend(dev: *const FfaiDevice) -> *mut c_char {
    if dev.is_null() {
        return std::ptr::null_mut();
    }
    let d = unsafe { &*dev };
    cstr(d.0.backend().as_str().to_string())
}

/// Human-readable device name (e.g. `"CUDA device (sm_121)"`).
#[unsafe(no_mangle)]
pub extern "C" fn ffai_device_name(dev: *const FfaiDevice) -> *mut c_char {
    if dev.is_null() {
        return std::ptr::null_mut();
    }
    let d = unsafe { &*dev };
    cstr(d.0.name().to_string())
}

/// Close a device handle opened by [`ffai_open_device`].
#[unsafe(no_mangle)]
pub extern "C" fn ffai_close_device(dev: *mut FfaiDevice) {
    if !dev.is_null() {
        unsafe { drop(Box::from_raw(dev)) };
    }
}
