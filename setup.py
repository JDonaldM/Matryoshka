import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

setuptools.setup(
    name="matryoshka",
    version="0.2.15",
    author="Jamie Donald-McCann",
    author_email="jamie.donald-mccann@port.ac.uk",
    description="Emulated halo model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JDonaldM/Matryoshka",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=requirements,
    python_requires='>=3.8',
)
