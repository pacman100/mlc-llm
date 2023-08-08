import tvm
import tvm._ffi.base

from . import libinfo, callback

# pylint: disable=line-too-long
_PYTHON_GET_STARTED_TUTORIAL_URL = "https://github.com/mlc-ai/notebooks/blob/main/mlc-llm/tutorial_chat_module_getting_started.ipynb"
# pylint: enable=line-too-long


def _load_mlc_llm_lib():
    """Load mlc llm lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    lib_name = "mlc_llm" if tvm._ffi.base._RUNTIME_ONLY else "mlc_llm_module"
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


# only load once here
if os.environ.get("SKIP_LOADING_MLCLLM_SO", "0") == "0":
    _LIB, _LIB_PATH = _load_mlc_llm_lib()


_get_delta_message = tvm.get_global_func("mlc.get_delta_message")


def get_delta_message(prev_message, curr_message):
    return _get_delta_message(prev_message, curr_message)
