import setuptools
from os.path import sep, split, abspath
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

here = split(abspath(__file__))[0]
datafiles = [here + sep + 'Coeffs16384Kaiser-quant.dat']

setuptools.setup(
    name="AAF",
    version="1.0",
    author="Anqi Li",
    author_email="li@astro.rug.nl",
    description="Apertif Anti-Aliasing Filter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apertif/aaf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe = False,
    data_files = [('share/aaf',
                  datafiles)],
    scripts=['aaf/aaf.py']
)
