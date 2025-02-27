# Setting up a new Python project

There is no canonical way: if your search the web, you will find many blogs and tutorials that explain one way of doing, each way has its pros and cons.
Here is another opinionated suggestion, using  `conda` as a Python package manager and a virtual environment manager.

1. Create a [new GitHub repository](https://github.com/new),
   - name it wisely, say `my_python_package`:
     - use lowercase and underscores,
     - see also the latest [trending Python GitHub repos](https://github.com/trending/python?).
   - write a short but meaningful description,
   - choose to make it public or private (this can be modified a posteriori)
   - initialize it with the following files:
      - `README.md`,
      - `.gitignore` and select the Python template,
      - `License`, for non sensitive projects consider a permissive license like the [MIT License](https://en.wikipedia.org/wiki/MIT_License).
   - invite collaborators: Settings/Manage access.

2. Clone the repository and navigate into the corresponding folder

    ```bash
    cd parent-directory-of-the-project
    git clone https://github.com/my_python_package.git
    cd my_python_package
    ```

3. Define a `conda` virtual environment for your project
   - consider naming the environment with the same name as the overall project
   - consider using an `environment.yml` file to be shared with your collaborators

4. Activate your `conda` virtual environment

   ```bash
   # cd my_python_package
   conda deactivate
   conda activate my_python_package  # a (my_python_package) prefix should appear
   ```

5. Structure your project files

    ```bash
    my_python_package  # project name
    ├── .gitignore  # list of files/folders not to track using git
    ├── LICENSE
    ├── README.md  # front/main page of the project
    ├── environment.yml  # conda virtual environment file
    ├── pyproject.toml  # project build and tools configuration
    ├── setup.cfg  # project meta-data, dependencies and tools configuration
    ├── setup.py  # useful for editable install with pip < 21.1
    ├── src  # source files of the project
    │   └── my_python_package  # package name
    │       ├── __init__.py  # make the directory a Python package
    │       ├── lab1  # sub-package name
    │       │   └── __init__.py
    │       └── lab2  # sub-package name
    │           └── __init__.py
    ├── tests  # unit-testing
    │   ├── __init__.py
    │   ├── lab1
    │   │   ├── __init__.py
    │   │   └── test_example.py
    │   └── lab2
    │       ├── __init__.py
    │       └── test_example.py
    ├── docs  # documentation generated by sphinx
    │   ├── Makefile
    │   ├── make.bat
    │   ├── conf.py  # documentation configuration file
    │   ├── index.rst  # documentation front page and structure
    │   ├── lab1
    │   │   └── index.rst
    │   └── lab2
    │       └── index.rst
    ├── notebooks  # sugar used at best to showcase your project
    │   ├── GUCKERT-GARDOIS-Lab1.ipynb
    │   └── GUCKERT-Lab2.ipynb
    ├── .vscode  # Visual Studio Code local configuration
    │   ├── workspace.code-snippets  # custom snippets
    │   ├── extensions.json  # recommended extensions
    │   ├── launch.json  # debugging configurations
    │   └── settings.json  # generic settings
    └── notes  # general or personal notes
        ├── note1.md
        └── note2.md
    ```

6. Define the project meta-data and dependencies in the [`setup.cfg`](../setup.cfg) file

7. Install the project in editable mode

    ```bash
    # cd my_python_package
    conda activate my_python_package  # a (my_python_package) prefix should appear
    pip install -e .
    ```

8. Start working, see also [`project-workflow.md`](./project-workflow.md).

## To go further

If you're happy with the resulting package templating process, you can think of converting the repository into a

- [GitHub template repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), or a
- [cookiecutter](https://cookiecutter-pypackage.readthedocs.io/en/latest/) package
