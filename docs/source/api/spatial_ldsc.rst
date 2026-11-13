Step 4: spatial_ldsc
====================

.. argparse::
   :module: gsMap.main
   :func: create_parser
   :prog: gsmap
   :path: run_spatial_ldsc

Checkpoint and resume
---------------------

Add ``--resume`` to the **first run** of ``gsmap run_spatial_ldsc`` or
``gsmap quick_mode`` to save each completed chunk/trait result. After an
interruption, rerun the same command with ``--resume``. Completed results are
reused before loading the remaining LD score chunks. Shared inputs are still
loaded once per invocation.

Checkpoints are stored under ``spatial_ldsc/.checkpoints/``. They are separated
by gsMap version, relevant parameters, the full trait list, and input file paths,
sizes and modification times. Changing these starts a fresh set of checkpoints.
``--chunk_range`` and the worker count may change when resuming. The final output
format is unchanged; without ``--resume``, chunk checkpoints are neither written
nor reused.

Checkpoints for the current inputs are deleted after all chunks and final output
files are successfully completed. Interrupted runs and partial ``--chunk_range``
runs keep their checkpoints for recovery.
