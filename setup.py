#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
from setuptools import find_packages
from setuptools import setup

""" See https://blog.ionelmc.ro/2014/05/25/python-packaging/ """


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

# Get the long description from the README file
long_description = read('README.md')

setup(
    name='genus',  # Required
    version='0.0.1',  # Required
    description='GENerative framework for the Unsupervised Segmentation of modular images.',  # Optional
    long_description=long_description,  # Optional
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    long_description_content_type='text/markdown',  # Optional (see note above)
    author='Luca D\'Alessio',  # Optional
    author_email='ldalessi@broadinstitute.org',  # Optional
    url='https://github.com/broadinstitute/genus.git',  # Optional
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    package_data={'': ['_default_params_CompositionalVae.json']},
    include_package_data=True,
    zip_safe=False,
    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='instance segmentation, machine learning, ML, unsupervised segmentation',  # Optional
    python_requires='>=3.5, <4',
    install_requires=['leidenalg >= 0.8',
                      'neptune-client >= 0.4.120',
                      'neptune-contrib',
                      'torch',
                      'dill >= 0.3.2',
                      'torchvision'],  # Optional
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'nameless = nameless.cli:main',
        ]
    },
)
