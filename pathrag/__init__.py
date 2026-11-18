import os
import platform

# torch and faiss-cpu each ship their own OpenMP runtime; on macOS both load
# and faiss segfaults. One OpenMP thread avoids it. Must run before either
# library is imported, so it lives here.
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")