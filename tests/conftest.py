import git
import pytest
import os

data_repo_names = ["recurrent-data", "transfer-data"]

@pytest.fixture(scope="session", autouse=True)
def setup():
    for data_repo_name in data_repo_names:
        if os.path.isdir(data_repo_name):
            print("Updating data repo...")
            git.Git(data_repo_name).pull()
        else:
            print("Fetching data repo...")
            data_repo_link = f"git://github.com/scikit-ika/{data_repo_name}.git"
            git.Git(".").clone(data_repo_link)
