from setuptools import setup, find_packages

setup(
    name='xrdfit',
    version='0.1',
    description='Automated fitting of XRD peaks using Pseudo-Voight fits',
    author='Peter Crowther',
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'dill',
                      'tqdm',
                      'scipy',
                      'lmfit',
					  'jupyter',
                      ],
    extras_require={"documentation_compilation": "sphinx"}
)