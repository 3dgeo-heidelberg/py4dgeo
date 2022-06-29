from skbuild import setup


setup(
    name="py4dgeo",
    version="0.4.0",
    author="Dominic Kempf",
    author_email="ssc@iwr.uni-heidelberg.de",
    description="Library for change detection in 4D point cloud data",
    long_description="",
    long_description_content_type="text/markdown",
    packages=["py4dgeo"],
    install_requires=[
        "dateparser",
        "laspy[lazrs]>=2.0,<3.0",
        "matplotlib",
        "numpy",
        "requests",
        "ruptures",
        "seaborn",
        "xdg",
    ],
    entry_points={
        "console_scripts": [
            "copy_py4dgeo_test_data=py4dgeo.util:copy_test_data_entrypoint"
        ]
    },
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
)
