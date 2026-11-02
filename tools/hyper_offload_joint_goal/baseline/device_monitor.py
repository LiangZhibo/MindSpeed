"""Read device-wide HBM via the installed DCMI ABI; values retain raw units."""
import argparse
import ctypes
import json
import time
from pathlib import Path


class HBM(ctypes.Structure):
    _fields_ = [('memory_size', ctypes.c_ulonglong), ('freq', ctypes.c_uint),
                ('memory_usage', ctypes.c_ulonglong), ('temp', ctypes.c_int),
                ('bandwidth_util_rate', ctypes.c_uint)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--stop-file', required=True)
    parser.add_argument('--interval', type=float, default=.05)
    parser.add_argument('--cards', default='6,7')
    args = parser.parse_args()
    lib = ctypes.CDLL('/usr/local/Ascend/driver/lib64/driver/libdcmi.so')
    lib.dcmi_init.restype = ctypes.c_int
    assert lib.dcmi_init() == 0
    lib.dcmi_get_device_hbm_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(HBM)]
    lib.dcmi_get_device_hbm_info.restype = ctypes.c_int
    cards = [int(x) for x in args.cards.split(',')]
    with open(args.output, 'w', buffering=1) as f:
        while not Path(args.stop_file).exists():
            t = time.monotonic()
            for card in cards:
                x = HBM()
                before = time.monotonic()
                code = lib.dcmi_get_device_hbm_info(card, 0, ctypes.byref(x))
                row = dict(time=time.time(), monotonic=before, card=card, code=code,
                           query_seconds=time.monotonic() - before,
                           **{k: getattr(x, k) for k, _ in HBM._fields_})
                f.write(json.dumps(row) + '\n')
            time.sleep(max(0, args.interval - (time.monotonic() - t)))


if __name__ == '__main__':
    main()
