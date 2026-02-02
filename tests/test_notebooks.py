import os

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS_DIR = "notebooks"
TEST_COLLECTION = "tests/test_collection.anki2"


def run_notebook(notebook_path: str) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    try:
        # Use notebooks dir for relative paths to work as expected
        # by the notebooks. Or we can set the resources path.
        client = NotebookClient(
            nb,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": NOTEBOOKS_DIR}},
        )

        print(f"\nExecuting notebook: {notebook_path}")
        # Use "Default" deck config for tests
        os.environ["DECK_CONFIG"] = "Default"
        # Use test collection for tests
        os.environ["ANKI_COLLECTION"] = os.path.abspath(TEST_COLLECTION)
        # Speed up simulations for tests
        os.environ["N_DAYS"] = "1"
        os.environ["REPEATS"] = "1"
        # Execute cells one by one to provide progress and prevent timeouts
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    client.execute_cell(cell, i)
                    print(f"  Cell {i + 1}/{len(nb.cells)} completed.")
    finally:
        pass


@pytest.mark.integration  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "notebook_name",
    [
        "Divergence_Exploration.ipynb",
        "Optimal_Retention.ipynb",
        "Review_Time_Analysis.ipynb",
    ],
)
def test_notebook_execution(notebook_name: str) -> None:
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook_name)
    run_notebook(notebook_path)
