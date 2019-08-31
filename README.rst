k-t BLAST
=========

Python implementation of several k-t BLAST-like MR reconstruction
algorithms.

Currently included modules are:

- UNFOLD
- k-t BLAST

Installation
============

Should be a quick pip install:

.. code-block:: bash

    pip install ktblast

Usage
=====

See examples module and doc strings -- both good resources for full
arguments list and usage.

.. code-block:: python

    # UNFOLD algorithm
    from ktblast import unfold

    sx, sy, st = kspace[:]
    recon = unfold(kspace, time_axis=-1)

    # k-t BLAST algorithm
    from ktblast import ktblast

    sx, sy, st = kspace[:]
    sx, sy, st = calib[:]
    recon = ktblast(kspace, calib, psi, time_axis=-1)


About
=====

I couldn't find any implementation of these algorithms that were
easy to get up and running in Python, so I decided to write my own.
