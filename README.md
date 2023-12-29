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
python runner.py prepare --N=32
python runner.py train --N=32
python runner.py evaluate --N=32 --name=2023-12-28_17-02-22
```