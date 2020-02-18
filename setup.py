# !/usr/bin/env python

from setuptools import setup,  find_packages
setup(
    name='ploteries',
    packages = find_packages('.', exclude=['test']),
    scripts = ['ploteries/bin/ploteries'],
    version='0.1.0',
    description='',
    author='Joaquin Zepeda',
    )
