/* Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
 * SPDX-License-Identifier: Apache-2.0
 *
 * C ABI for the shared Butter Rust engine. Swift (and any C caller) links
 * libwh_butter_ffi and drives the same Device/ops layer the CUDA backend uses.
 * Strings returned by butter_* are heap-allocated; free them with
 * butter_string_free. */
#ifndef BUTTER_H
#define BUTTER_H

typedef struct ButterDevice ButterDevice;

char *butter_version(void);
/* Comma-separated list of backends compiled into this build. */
char *butter_compiled_backends(void);
void        butter_string_free(char *s);

/* Open the first live device (NULL if none — e.g. on Apple, where the
 * Swift-native Metal engine is the primary path). Close with
 * butter_close_device. */
ButterDevice *butter_open_device(void);
char *butter_device_backend(const ButterDevice *dev);
char *butter_device_name(const ButterDevice *dev);
void        butter_close_device(ButterDevice *dev);

#endif /* BUTTER_H */
