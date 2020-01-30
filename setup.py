from setuptools import setup, Extension
from setuptools import find_packages

import jointtsmodel

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="jointtsmodel",
        version="1.2",
        description="jointtsmodel - library of joint topic-sentiment models",
        long_description=long_description,
        long_description_content_type='text/markdown',
        author="Ayan Sengupta",
        author_email="ayan.sengupta007@gmail.com",
        url="https://github.com/victor7246/jointtsmodel",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=['pandas','numpy','scipy','nltk','sklearn']
    )
