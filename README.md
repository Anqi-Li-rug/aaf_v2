# Apertif Anti-Aliasing Filter

This code will apply an anti-aliasing filter to Apertif data. The code is written by Anqi Li, based on work by Stefan Wijnholds and Sebastiaan van der Tol.

To use the code, use the stand-alone mode:

```bash
aaf.py --help
usage: aaf.py [-h] [-t TOLERANCE]  msname

Apertif Anti-aliasing filter.

positional arguments:
  msname                Name of Measurement Set

optional arguments:
  -h, --help            show this help message and exit
  -t TOLERANCE, --tolerance TOLERANCE
                        Filter response below this limit will be ignored
```

Also, it can be used from python:

To run the code in python: 
```python
from aaf import antialias_ms
antialias_ms('WSRTA18017026_B00.MS/', 0.00001)
```
