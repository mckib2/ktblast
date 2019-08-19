k-t BLAST
=========

Python implementation of several k-t BLAST-like MR reconstruction
algorithms.

Usage
=====

.. code-block::python

    # UNFOLD algorithm
    from ktblast import unfold
    sx, sy, st = kspace[:]
    recon = unfold(kspace)

    # k-t BLAST algorithm
    from ktblast import ktblast

About
=====

I couldn't find any implementation of these algorithms that were
easy to get up and running in Python, so I decided to write my own.
