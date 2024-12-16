from typing import TypeAlias, Callable, Union
import numpy as np

ArrayLike: TypeAlias = Union[np.ndarray, list, tuple]
UFunc: TypeAlias = Callable[[np.ndarray], np.ndarray]
