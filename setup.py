from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      install_requires=['scipy', 'numpy', 'pandas', 'sympy', 'sklearn', 
                       		    'statsmodels'],
      include_package_data=True,
      package_data={'': ['examples/*.csv', 'examples/*.npy']},
      name="semopy",
      version="2.1.4",
      author="Georgy Meshcheryakov",
      author_email="metsheryakov_ga@spbstu.ru",
      description="Structural Equation Modeling Optimization in Python.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://semopy.com",
      packages=find_packages(),
      python_requires=">=3.7",
      classifiers=[
              "Programming Language :: Python :: 3.7",
	      "Programming Language :: Python :: 3.8",
	      "Programming Language :: Python :: 3.9",
              "License :: OSI Approved :: MIT License",
	      "Development Status :: 5 - Production/Stable",
	      "Topic :: Scientific/Engineering",
              "Operating System :: OS Independent"])
