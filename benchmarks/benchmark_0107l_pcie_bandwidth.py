"""Bounded pinned-memory PCIe copies only; no model imports or GPU kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import cupy as cp
import numpy as np


WARMUP = 2
REPEATS = 5


def pci_metadata(bus_id):
    endpoint = Path('/sys/bus/pci/devices') / bus_id.lower()
    result = []
    if endpoint.exists():
        for node in (endpoint.resolve(), *endpoint.resolve().parents):
            if not (node / 'vendor').exists():
                continue
            entry = {'pci_bus_id': node.name}
            for field in ('vendor', 'device', 'numa_node', 'current_link_speed',
                          'current_link_width', 'max_link_speed', 'max_link_width'):
                try:
                    entry[field] = (node / field).read_text().strip()
                except OSError:
                    pass
            result.append(entry)
    return result


def benchmark_size(mib):
    nbytes = mib * 1024 * 1024
    source_memory = cp.cuda.alloc_pinned_memory(nbytes)
    destination_memory = cp.cuda.alloc_pinned_memory(nbytes)
    source = np.frombuffer(source_memory, dtype=np.uint8, count=nbytes)
    destination = np.frombuffer(destination_memory, dtype=np.uint8, count=nbytes)
    # Identical exact bytes across runtimes, sizes, and devices, without RNG state.
    pattern = ((np.arange(65536, dtype=np.uint32) * 37 +
                np.arange(65536, dtype=np.uint32) // 251 + 17) % 256).astype(np.uint8)
    source.reshape(-1, pattern.size)[:] = pattern
    destination.fill(0)
    device_memory = cp.cuda.alloc(nbytes)
    stream = cp.cuda.get_current_stream()
    runtime = cp.cuda.runtime

    def h2d():
        runtime.memcpyAsync(device_memory.ptr, source.ctypes.data, nbytes,
                            runtime.memcpyHostToDevice, stream.ptr)

    def d2h():
        runtime.memcpyAsync(destination.ctypes.data, device_memory.ptr, nbytes,
                            runtime.memcpyDeviceToHost, stream.ptr)

    # Initialize device storage and establish the exact-byte round trip before timing.
    h2d()
    d2h()
    stream.synchronize()
    if not np.array_equal(source, destination):
        raise AssertionError(f'{mib} MiB initial round-trip byte mismatch')

    measurements = {}
    for name, copy in (('H2D', h2d), ('D2H', d2h)):
        for _ in range(WARMUP):
            copy()
            stream.synchronize()
        samples = []
        for _ in range(REPEATS):
            stream.synchronize()
            start = time.perf_counter_ns()
            copy()
            stream.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e9)
        median = statistics.median(samples)
        measurements[name] = {
            'seconds_samples': samples,
            'median_seconds': median,
            'median_ms': median * 1000,
            'GB_per_second': nbytes / median / 1e9,
        }
        # A fresh readback validates the bytes after each direction's measurements.
        destination.fill(0)
        d2h()
        stream.synchronize()
        if not np.array_equal(source, destination):
            raise AssertionError(f'{mib} MiB {name} round-trip byte mismatch')

    return {
        'MiB': mib,
        'bytes': nbytes,
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'roundtrip_sha256': hashlib.sha256(destination).hexdigest(),
        'roundtrip_bytes_equal': True,
        'directions': measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties['name']
    if isinstance(name, bytes):
        name = name.decode()
    bus_id = device.pci_bus_id
    report = {
        'device_name': name,
        'device_id': device.id,
        'device_pci_bus_id': bus_id,
        'pci_path_endpoint_to_root': pci_metadata(bus_id),
        'cupy_version': cp.__version__,
        'numpy_version': np.__version__,
        'is_hip': bool(cp.cuda.runtime.is_hip),
        'runtime_version': cp.cuda.runtime.runtimeGetVersion(),
        'driver_version': cp.cuda.runtime.driverGetVersion(),
        'python_version': platform.python_version(),
        'backend_environment': os.environ.get('MINI_LLM_BACKEND'),
        'warmup_per_size_per_direction': WARMUP,
        'repeats_per_size_per_direction': REPEATS,
        'timing': 'perf_counter_ns CPU wall; current stream synchronized before and after each copy',
        'allocation': 'alloc_pinned_memory + numpy.frombuffer; cuda.alloc device buffer; initialized before timing',
        'bandwidth_units': 'decimal GB/s = bytes / seconds / 1e9',
        'results': [benchmark_size(mib) for mib in (16, 128)],
    }
    text = json.dumps(report, indent=2)
    with output.open('x') as handle:
        handle.write(text + '\n')
    print(text, flush=True)


if __name__ == '__main__':
    main()
