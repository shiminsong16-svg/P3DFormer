import os
import sys
import glob

# Ensure the compiled CUDA extension `attention_rpe_ops_cuda` is importable.
# When built locally, it may reside under `build/lib.*`.
_here = os.path.dirname(__file__)
_build_lib_candidates = glob.glob(os.path.join(_here, 'build', 'lib.*'))
for _p in _build_lib_candidates:
	if _p not in sys.path:
		sys.path.append(_p)

# Optional sanity import: don't hard-fail here to allow package import
# even if extension isn't present yet (e.g., CPU-only env).
try:
	import attention_rpe_ops_cuda  # noqa: F401
except Exception:
	# Extension might not be built yet; functions module will raise on use.
	pass
