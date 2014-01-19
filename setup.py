#!/usr/bin/env python2
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="learnC",
    ext_modules=[
        Extension("learnC", ["learn.cpp"],
        libraries = ["boost_python"])
    ])
