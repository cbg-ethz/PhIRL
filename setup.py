import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


setup(
    name="PhIRL",
    version="0.1",
    license="MIT",
    description="Inverse Reinforcement Learning for phylogenetic trees.",
    long_description=re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
        "", read("README.md")
    ),
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[],  # TODO(Pawel): We'll need to specify those or add the requirements.txt file
    setup_requires=[
        "pytest-runner",
    ],
)
