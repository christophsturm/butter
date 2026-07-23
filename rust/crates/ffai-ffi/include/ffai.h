/* Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
 * SPDX-License-Identifier: Apache-2.0
 *
 * C ABI for the shared FFAI Rust engine. Swift (and any C caller) links
 * libiron_ffi and drives the same Device/ops layer the CUDA backend uses.
 * Strings returned by iron_* are heap-allocated; free them with
 * iron_string_free. */
#ifndef FFAI_H
#define FFAI_H

typedef struct FfaiDevice FfaiDevice;

char *iron_version(void);
/* Comma-separated list of backends compiled into this build. */
char *iron_compiled_backends(void);
void        iron_string_free(char *s);

/* Open the first live device (NULL if none — e.g. on Apple, where the
 * Swift-native Metal engine is the primary path). Close with
 * iron_close_device. */
FfaiDevice *iron_open_device(void);
char *iron_device_backend(const FfaiDevice *dev);
char *iron_device_name(const FfaiDevice *dev);
void        iron_close_device(FfaiDevice *dev);

#endif /* FFAI_H */
