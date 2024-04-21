# !/usr/bin/env python

from setuptools import setup, find_packages

with open("readme.md") as f:
    long_description = f.read()


setup(
    name="ploteries",
    packages=find_packages(".", exclude=["tests"]),
    scripts=["ploteries/bin/ploteries"],
    version="0.1.2",
    description="Plotting for ML training",
    keywords=["plots", "tensorboard", "machine learning", "training"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zepedaj/soleil",
    author="Joaquin Zepeda",
    install_requires=[
        "pytest",
        "sphinx",
        "enum_tools",
        "numpy",
        "sqlalchemy==1.4.46",
        "climax",
        "dash",
        "dash_daq",
        "gunicorn",
        "jztools>=0.1.5",
        "pandas",
        "rich",
    ],
)
