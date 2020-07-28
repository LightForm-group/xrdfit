from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='xrdfit',
    version='1.0.0',
    description='Automated fitting of XRD peaks using Pseudo-Voight fits',
    author='Peter Crowther, Christopher Daniel',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LightForm-group/xrdfit",
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'dill',
                      'tqdm',
                      'lmfit',
                      'notebook',
                      'ipywidgets'
                      ],
    extras_require={"documentation_compilation": "sphinx"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)