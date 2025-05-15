import setuptools

required_packages = [
    "matplotlib",
    "numpy",
    "pandas",
    "openpyxl==3.1.5",
    "Sphinx==7.2.6",
    "pylint",
    "sphinx-book-theme == 1.1.2",
    "pytest",
    "scipy==1.14.0",
    "termcolor",
]

setuptools.setup(
    name="teva",
    version='0.0.5',
    description="Tandem Evolutionary Algorithm",
    author="Transcend Engineering & Technologies, LLC",
    author_email="bleavitt@transcendengineering.com",
    packages=setuptools.find_packages(),
    install_requires=required_packages,
    include_package_data=True,  # Include package data such as data files and templates
    setup_requires=['wheel'],  # Add this line to enable wheel distribution
)