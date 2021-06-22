# !/usr/bin/env python

from setuptools import setup,  find_packages
setup(
    name='ploteries',
    packages=find_packages('.', exclude=['tests']),
    scripts=['ploteries/bin/ploteries', 'ploteries2/bin/ploteries2'],
    version='0.1.0',
    description='',
    author='Joaquin Zepeda',
    install_requires=[
        # Testing
        'pytest',
        # !!!! sudo apt install graphviz  !!!!
        # Documentation
        'sphinx', 'enum_tools',  # 'sphinx-toolbox',
        # Algorithm
        'numpy', 'sqlalchemy',
        # Extra
        'climax', 'dash', 'dash_daq', 'gunicorn',
    ],
)
