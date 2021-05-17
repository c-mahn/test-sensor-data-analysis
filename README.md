# test-sensor-data-analysis
This is a Python script for anaysing sensor information

# Notizen

## Code-Beispiele

Lineare interpolation von Lücken in Messreihen
```
import numpy as np
from scipy import interpolate

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B
```

## Links

[NumPy documentation](https://numpy.org/doc/stable/reference/index.html)

[SciPy documentation](https://docs.scipy.org/doc/scipy/reference/)

[SciPy signal processing documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)

[Low pass filter example](https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units)

[Linear regression example](https://devarea.com/linear-regression-with-numpy/)

[Python datetime documentation](https://docs.python.org/3/library/datetime.html)
