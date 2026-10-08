# Tested POT wheel

`pot-0.9.6.post1-cp314-cp314-macosx_26_0_arm64.whl` is the exact source-built wheel used by the corrected native macOS arm64 checks. Its SHA256 and upstream source identity are recorded in `../core-cp314-macos-arm64.lock.manifest.json` and the closed requirements lock. The wheel contains POT's MIT license. Upstream: https://github.com/PythonOT/POT.

`otool -L` confirms only system libc++ and libSystem dependencies. This artifact targets CPython3.14 and macOS26 arm64; compatibility with other interpreters/platforms is not claimed. Other locked wheels are reproducibly acquired from their distributions with pip hash verification. GitHub's documented `macos-26` label selects arm64: https://docs.github.com/en/actions/reference/runners/github-hosted-runners.

`hic_straw-1.3.1-cp314-cp314-macosx_26_0_arm64.whl` preserves the native wheel used by the executed remote Hi-C range reads. Its adjacent manifest records the exact PyPI source SHA256, wheel SHA256, ABI and system libcurl/zlib/libc++ dependencies. Import and real range reads passed on the host; a fresh locked full-controls installation remains a separate acceptance gate. The package is distributed upstream under the MIT license (PyPI source metadata).
