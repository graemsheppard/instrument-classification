# Instrument Classification

The goal of this program is to label multiple instuments present in an audio file at a point in time.

## Training data

Training data can be downloaded [here](https://www.upf.edu/web/mtg/irmas) and should be places in `src/training_data` such that the subdirectories of `src/training_data` are "cel, cla, flu, etc..."

## Development Environment
 - Keep venv files in folder named venv or env (gitignored) `python3 -m venv venv`
 - Activate venv before installing packages and running code `source venv/bin/activate` (Unix) `venv/Scripts/activate` (Windows)
 - Install packages `pip install -r requirements.txt`
 - When updating `requirements.txt`, use `pipreqs` (`pip install pipreqs)

## Usage
`py main.py -p <path>`
Use the optional flag `-b` to force rebuilding the model.