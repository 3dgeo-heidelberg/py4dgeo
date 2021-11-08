from skbuild import setup
from setuptools import find_packages


setup(
    name="py4dgeo",
    version="0.0.1",
    author="Dominic Kempf",
    author_email="ssc@iwr.uni-heidelberg.de",
    description="Library for change detection in 4D point cloud data",
    long_description="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xdg",
    ],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    cmake_args=[
        "-DBUILD_DOCS=OFF",
        "-DBUILD_TESTING=OFF",
    ],
    package_dir={"": "src"},
    cmake_install_dir="src/py4dgeo",
    include_package_data=True,
)
