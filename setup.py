from pathlib import Path

from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup

project_name = "GooseEPM"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=project_name,
    description="Elasto Plastic Model",
    long_description=long_description,
    version=get_version(),
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    url=f"https://github.com/tdegeus/{project_name}",
    packages=find_packages(exclude=["tests"]),
    package_dir={"": "python"},
    cmake_install_dir=f"python/{project_name}",
    cmake_minimum_required_version="3.13...3.21",
)