from setuptools import setup, find_packages

setup(
    name="CAIN",
    version="0.1.0",
    description="CAvity investigation Navigator, a tool for functional protein cavity analysis.",
    packages=find_packages(include=["CAIN*"]),
    author="Jiahui Zhang",
    author_email="jzhang71@ncsu.edu",
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'scikit-learn',
    ]
)
