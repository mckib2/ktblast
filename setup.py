'''Setup.py
'''

from setuptools import setup, find_packages

setup(
    name='ktblast',
    version='0.1.0',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/ktblast',
    license='GPLv3',
    description=(
        'Python implementation of the k-t BLAST algorithm.'),
    long_description=open('README.rst').read(),
    keywords=(
        'mri BLAST parallel-imaging image-reconstruction python'),
    install_requires=[
        "numpy>=1.16.4",
        "matplotlib>=2.2.4",
        "phantominator>=0.3.2",
        "scikit-image>=0.15.0",
        "tqdm>=4.32.2",
    ],
    python_requires='>=3.5',
)
