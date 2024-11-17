import os

from setuptools import setup, find_namespace_packages

DIST_NAME = "FlexibleQECSim"
PACKAGE_NAME = "FlexibleQECSim"

REQUIREMENTS = [
    "numpy",
    "stim",
    "pymatching",
]
EXTRA_REQUIREMENTS = [

]
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

setup(
    name=DIST_NAME,
    version=version_str,
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/JiakaiW/FlexibleQECSim",
    author="Jiakai Wang",
    author_email="jwang2648@wisc.edu",
    license="Apache 2.0",
    packages=find_namespace_packages(exclude=['notebooks']),
    install_requires=REQUIREMENTS,
    extras_require=None,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    keywords="surface code erasure decoding",
    python_requires=">=3.7",
    project_urls={
        # "Documentation": "https://github.com/JiakaiW/FlexibleQECSim",
        "Source Code": "https://github.com/JiakaiW/FlexibleQECSim",
    },
    include_package_data=True,
)
