import git
import pytest
import os

data_repo_name = "recurrent-data"
data_repo_link = f"git://github.com/scikit-ika/{data_repo_name}.git"

@pytest.fixture(scope="session", autouse=True)
def setup():
    if os.path.isdir(data_repo_name):
        print("Updating data repo...")
        git.Git(data_repo_name).pull()
    else:
        print("Fetching data repo...")
        git.Git(".").clone(data_repo_link)
