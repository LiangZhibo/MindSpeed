#!/usr/bin/env bash
set -euo pipefail
: "${ASCEND_HOME_PATH:?Source the target CANN set_env.sh first}"
source_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
build_dir="${1:?Usage: bash build.sh ABSOLUTE_BUILD_DIRECTORY}"
[[ "$build_dir" == /* ]] || { echo 'Use an absolute build directory' >&2; exit 2; }
mkdir -p "$build_dir"
# This SDK links some intermediate objects in place; rebuilding the same tree
# can try to link an already merged object. Keep each build isolated.
kernel_dir="$(mktemp -d "$build_dir/kernel.XXXXXX")"
cmake -S "$source_dir" -B "$kernel_dir"
cmake --build "$kernel_dir" -j 4
kernel_archive="$(find "$kernel_dir" -name libsplitzip_kernel.a -print -quit)"
test -n "$kernel_archive"
g++ -std=c++17 -O2 -fPIC -shared \
  "$source_dir/bridge.cpp" "$source_dir/vendor/splitzip_v2.cpp" \
  -I "$source_dir/vendor" -I "$ASCEND_HOME_PATH/include" \
  "$kernel_archive" -L "$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -Wl,--no-undefined -Wl,--start-group -lascendcl -lnnopbase -lc_sec -lruntime \
  -lprofapi -lascendalog -lunified_dlog -lmmpa -ldl -lpthread -Wl,--end-group \
  -o "$build_dir/libmindspeed_splitzip.so"
echo "$build_dir/libmindspeed_splitzip.so"
