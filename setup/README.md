# Optional Setup Instructions

[Toc]

<br>

For detailed instructions on how to set up your machine and use the project code in different ways, please refer to the [official documentation](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup).

Here I will introduce the method I recommend, which is also the method I used to perfect this project.

## System Info

- OS: macOS Sequoia 15.0.1
- Chip: Apple M2 Max
- Memory: 32 GB
- Python: 3.9.18

> [!NOTE]
> For the parts of subsequent chapters that require powerful computing power, A100-80G will be used to run.

## Environment Setup

I use pyenv to manage Python versions and Poetry to manage dependencies. If you are not familiar with these tools, you can refer to the following instructions.

### pyenv

1. Install pyenv

    ```bash
    brew install pyenv
    ```

2. Install python 3.9.18

    ```bash
    pyenv install 3.9.18
    ```

3. Set the local python version to 3.9.18

    ```bash
    cd LLMs-from-scratch
    pyenv local 3.9.18
    ```

### Poetry

1. Install Poetry
    If you have pipx installed, you can install Poetry with pipx. If you don't have pipx installed, you can install Poetry with the official installation script.

    ```bash
    pipx install poetry
    ```

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Set the python version to 3.9.18

    ```bash
    poetry env use 3.9.18
    ```

3. Install dependencies

    ```bash
    poetry install
    ```

Because we use pyenv and poetry to manage the environment, so we don't need to double check the python version and dependencies.

## Format

### Install the tools

I use black and isort to format the code. You can use the following command to format the code.

```bash
pipx install black isort
```

Then I use pre-commit and commitizen to manage the commit. You can use the following command to install them.

```bash
pipx install pre-commit commitizen
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
