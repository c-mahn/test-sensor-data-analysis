# test-sensor-data-analysis
This is a Python script for anaysing sensor information

# Notizen

## Code-Beispiele

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