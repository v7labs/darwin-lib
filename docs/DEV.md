# Development Environment
This doesn't represent the only way to develop on darwin-py, but does represent an easy and configurable way to manage things like underlying dependencies and python versions
## Devcontainer - One click approach
Just use the devcontainer config, either in Github codespaces or in using the [VSCode extension](https://code.visualstudio.com/docs/devcontainers/tutorial). 

NB the config is [here](../.devcontainer/devcontainer.json)

Remember to activate the _venv_ after the container boots `source .venv/bin/activate`
## Manual setup
### Shell environment
No requirement for any particular shell, [zsh](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH) + [oh my zsh](https://ohmyz.sh/) is a good setup commonly used, but whatever environment you use make sure to install the recommended alias's and path exports that the below systems require for your particular shell environment, particularly pertinent for poetry which has an added step that it prints to console but isn't included on the webpage. 
### Pyenv
Pyenv manages system python versions, install instructions can be found [here](https://github.com/pyenv/pyenv). 
After installation of pyenv, install a python version that is compatible with darwin-py (3.9-3.12 as of writing)

`pyenv install 3.10`

If the command `pyenv` isn't recognized, it hasn't installed to your shell environemnt config file correctly .zshrc .bashrc etc. 
### Poetry
Poetry manages project level dependencies and local python versions. Install instructions [here](https://python-poetry.org/docs/). Make sure to follow the printed instructions and add the path to your shell environment, if running the command `poetry --version` after installation doesn't work, it means your path hasn't been updated

### New Folder Setup
To Start from scratch and get a development/QA environemnt setup. This process means you will have a fresh python version with only the dependencies required by darwin-py that is uncorrupted by other packages installed on the system python
- clone darwin py repo
- navigate to downloaded repo
- Set pyenv to use a local `pyenv local <version>` eg `pyenv local 3.10`
- Create local environment with poetry `poetry shell`
- Install dependencies `poetry install --all-extras`

Pyenv + Poetry here get used in conjuction, with pyenv telling the system whenever `python` is called in a folder that has been set with `pyenv local <version>` that it should use that local version. Poetry is then set to prefer that local version of python, and to create a per project copy of python to use, it will clone `<version>` into a .venv folder locally, and install dependencies there. If new environment is required, run `rm -rf .venv` while inside the project folder, set a new pyenv version if needed and re-run poetry commands 

### Subsequent Uses
Once a folder is setup, it can easily be reused
- navigate to folder
- run `poetry shell`
- execute any commands as normal eg `python -m darwin.cli ...`
- once complete, close terminal or manually exit shell via `exit` in terminal

Can also force poetry commands without being in a shell environment by prepending the command with `poetry run ...` for example

`poetry run python -m darwin.cli`

###  Testing
To run unit tests locally:
```
pytest
```

To run end-to-end tests locally, copy `e2e_tests/.env.example` to `.env` and populate all variables. Then run:
```
pytest e2e_tests
```

### Code Formatting and Linting
The project uses two main tools for code quality:

1. **Black** - The uncompromising code formatter
   - Automatically formats Python code to a consistent style
   - Run locally before committing:
   ```
   black .
   ```
   - CI will check formatting with `black --check`

2. **Ruff** - An extremely fast Python linter
   - Enforces code style and catches potential errors
   - Run locally:
   ```
   ruff check .
   ```

Both tools are automatically run in CI/CD pipelines for all Python files changed in pull requests. The workflow will:
- Check code formatting with Black
- Run Ruff linting checks
- Fail the build if any issues are found

To ensure your code passes CI checks, you can run these tools locally before pushing:
```bash
# Format code
black .

# Run linter
ruff check .
```

For VS Code users, it's recommended to enable format-on-save with Black and install the Ruff extension for real-time linting feedback.

## Useful Aliases
Aliases can be helpful for testing and development. Add them to your shell configuration file .bashrc .zshrc etc for ease of use and development
```
DARWIN_PY_DEV="$HOME/Development/darwin-py"
alias dpy="poetry run python -m darwin.cli"
alias dpyt="poetry run python -m pytest -W ignore::DeprecationWarning"
alias dpydb="poetry run python -m debugpy --listen 5678 --wait-for-client $DARWIN_PY_DEV/darwin/cli.py"
```

- dpy -> quick way to run darwin
- dpyt -> calls pytest
- dpydb -> creates a remote attach debugging instance for vscode to attach to