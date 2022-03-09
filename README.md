## Gunfolds Documentation

This repository contains the documentation of Gunfolds

## Acknowledgment

This work was supported by  NSF IIS-1318759 grant.

## Setting up Gunfolds Docs
This repository will create a documentation project on Gunfolds by using Sphinx Documentation.

You will fork this repository and the clone to your system to execute the documentation. After cloning, you need to checkout the master branch before proceeding further. 

docs/source/ 
Directory holding all the spinx doucmented file with the rst extension. 
docs/html/ 
Directory holding all the html doucmented file of the documentation.  


### Quickstart Sphinx
This installs sphinx on your machine and helps run the sphinx documentation
```
  $ pip install sphinx
```
This theme used by documentation is read the docs theme, therefore, it is installed to execute the documentation. 
```
  $ pip install sphinx_rtd_theme
```
To start the sphinx execution, we need to first setup virtual environment (If this repository is forked, virtual env is not needed to set up externally). We just need to activate the virtual environment before proceeding. The execution command is stated below: 
```
  $ cd gunfoldsdocs
  $ source gunfoldsdocsenv/bin/activate
```
Once the enviornment is activated, we need to cd to docs directory and run the html files. Therefore, 
```
  $ cd docs
  $ make html
```
In the end, open docs/build/html/index.html in the browser


