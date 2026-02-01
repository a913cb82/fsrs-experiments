import os

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS_DIR = "notebooks"
TEST_COLLECTION = "tests/test_collection.anki2"


def run_notebook(notebook_path: str) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # We need to be in the notebooks directory for relative paths to work as expected
    # by the notebooks. Or we can set the resources path.
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOKS_DIR}},
    )

    # We need to make sure the collection.anki2 exists in the parent dir of notebooks/
    # because notebooks/Review_Time_Analysis.ipynb expects ../collection.anki2
    # For testing purposes, we can link the test collection
    link_path = "collection.anki2"
    created_link = False
    if not os.path.exists(link_path):
        os.symlink(TEST_COLLECTION, link_path)
        created_link = True

    try:
        client.execute()
    finally:
        if created_link:
            os.remove(link_path)


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
