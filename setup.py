# !/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="ploteries",
    packages=find_packages(".", exclude=["tests"]),
    scripts=["ploteries/bin/ploteries"],
    version="0.1.0",
    description="",
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
