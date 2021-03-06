#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []
with open('requirements_dev.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Pradip Solanki",
    author_email='solankipms@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'face_recognition_and_drowsiness_detection=face_recognition_and_drowsiness_detection.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='face_recognition_and_drowsiness_detection',
    name='face_recognition_and_drowsiness_detection',
    package_dir={"": "src"},
    packages=['face_recognition_and_drowsiness_detection'],
    setup_requires=setup_requirements,
    py_modules=['face_recognition_and_drowsiness_detection'],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Pradip240/face_recognition_and_drowsiness_detection',
    version='0.1.1',
    zip_safe=False,
)
