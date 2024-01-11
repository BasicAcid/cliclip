# cliclip

CLI tool that uses the [CLIP](https://github.com/openai/CLIP) model
from OpenAI to search images on your disk.

Support Cuda and CPU.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```
usage: cliclip.py [-h] -d DIRECTORY -p PROMPT [-t TOP] [-s]

Search for images based on textual prompts using CLIP.

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Directory containing the images.
  -p PROMPT, --prompt PROMPT
                        Text prompt.
  -t TOP, --top TOP     Number of top scored images to display.
  -s, --score           Display score along output.
```
