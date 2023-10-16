@echo off
call conda activate hynet
black ..\hynet
isort ..\hynet
nbqa black ..
nbqa isort ..
call conda deactivate