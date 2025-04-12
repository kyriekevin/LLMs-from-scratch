# Optional Setup Instructions

- [Optional Setup Instructions](#optional-setup-instructions)
  - [System Info](#system-info)
  - [Environment Setup](#environment-setup)
    - [UV](#uv)
  - [Format](#format)
    - [Install the tools](#install-the-tools)
    - [Usage](#usage)

For detailed instructions on how to set up your machine and use the project code in different ways, please refer to the [official documentation](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup).

Here I will introduce the method I recommend, which is also the method I used to perfect this project.

## System Info

- OS: macOS Sequoia 15.0.1
- Chip: Apple M2 Max
- Memory: 32 GB
- Python: 3.11

> [!NOTE]
> For the parts of subsequent chapters that require powerful computing power, A100-80G will be used to run.

## Environment Setup

Use uv as the python version and third-party library management tool.

### UV

For Mac users, you can use homebrew to install uv, like `brew install uv`

After installing uv, first use uv to specify the python version and create a python project.

```bash
uv init
uv python pin <version>
uv venv --python <version>
```

Then you can use uv to install the required third-party libraries. If the Python version is not downloaded, uv will automatically download it.

```bash
uv add <package>
```

Finally, you can use `uv run xxx.py` to run the Python file, or use Jupyter to run the code.

## Format

### Install the tools

I use ruff as a code standardization tool for Python. You can use uv to install ruff globally, refer to the following command.

```bash
uv tool install ruff
```

And use pre-commit for checking before code submission, and commitizen for standardizing git commit information. You can use uv to install both two tools, refer to the following command.

```bash
uv tool install pre-commit
uv tool install commitizen
```

After that, you can use the following command to initialize the pre-commit.

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

### Usage

You can use the following command to format the code.

```bash
git add <file>
pre-commit run --all-files
```

Then all the code will be formatted.
