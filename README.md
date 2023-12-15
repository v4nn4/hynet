# hynet

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Training LeNet-5 on Armenian script.

## Installation

```
conda env create -n hynet -f environment.yml
```

## Evaluation

```
conda activate hynet
python runner.py prepare --N=56
python runner.py train --N=56 --batches=16
python runner.py evaluate --batches=16
```