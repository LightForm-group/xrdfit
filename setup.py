from setuptools import setup, find_packages

setup(
    name='xrd_fit',
    version='0.1',
    description='A package for automating the fitting of XRD peaks using Pseudo-Voight fits.',
    author='Peter Crowther',
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'dill',
                      'tqdm',
                      'scipy',
                      'lmfit'
                      ]
)