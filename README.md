# time-series-with-Konrad
Time series with Konrad - the code

If importing SciPy on macOS fails with `__thread_bss` / `offset field is not
zero`, the installed wheel contains a malformed Mach-O header
([upstream issue](https://github.com/scipy/scipy/issues/25635)). With the current
SciPy 1.15.3 environment, apply the local workaround from the project directory:

```sh
.venv/bin/python scripts/repair_scipy_macos.py
```

Restart the notebook kernel afterward. The script corrects only invalid
thread-local zero-fill offsets, re-signs the affected libraries locally, and
keeps originals beside them as `*.so.before-macos-repair`. Package versions are
unchanged. Reinstalling SciPy replaces the workaround; rerun it if the same
error returns. This is a local compatibility workaround, not an upstream fix.
