from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      install_requires=['scipy', 'numpy', 'pandas', 'sympy', 'sklearn', 
                        'statsmodels'],
      include_package_data=True,
      package_data={'': ['examples/*.csv']},
      name="semopy",
      version="2.0.5",
      author="Georgy Meshcheryakov",
      author_email="metsheryakov_ga@spbstu.ru",
      description="Structural Equation Modeling Optimization in Python.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://semopy.com",
      packages=find_packages(),
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	      "Development Status :: 3 - Alpha",
              "Operating System :: OS Independent"])
