import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebooks_in_directory(directory):
    for file in os.listdir(directory):
        if file.endswith(".ipynb"):
            notebook_path = os.path.join(directory, file)
            execute_notebook(notebook_path)

def execute_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {e}")


if __name__ == "__main__":
    current_directory = os.getcwd()
    print(current_directory)
    # execute_notebooks_in_directory(current_directory)
