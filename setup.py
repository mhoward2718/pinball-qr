#!/usr/bin/env python

"""The setup script."""
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import site

site.ENABLE_USER_SITE = True
# I have to fix this for unit tests to work because they test files
# that import native methods
# See this link for how to package native code in installer
# https://numpy.org/doc/stable/f2py/distutils.html

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'statsmodels', 'scipy', 'scikit-learn']

test_requirements = ['pytest>=3']

setup(
    author="Michael Howard",
    author_email='mah38900@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="The fastest and most accurate method to train quantile regression models in Python",
    entry_points={
        'console_scripts': [
            'pinball=pinball.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pinball',
    name='pinball',
    packages=find_packages(include=['pinball', 'pinball.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mah38900/pinball',
    version='0.0.1',
    zip_safe=False,
    ext_modules=[
        Extension(name="pinball_native", sources=["fortran/rqbr.f"])
    ]
)
