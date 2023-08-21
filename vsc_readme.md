# Overview

Here are some notes for using the VSC (Vienna Scientific Cluster) a high performance cluster which uses SLURM for job management. Most information can be found in their wiki. So the notes here are a cheatsheet or also provide some more specific information on the setup.

# Python Environment / Dependencies

- VSC uses conda for managing Python packages (not PIP)
- Documentation available here: https://wiki.vsc.ac.at/doku.php?id=doku:python
- This repository provides `conda.yml` with all conda dependencies

To install run:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"

conda env create -n env-oembeddings --file oembeddings/conda.yml
```

This takes a while!

