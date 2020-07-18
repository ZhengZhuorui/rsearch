from setuptools import setup, find_packages
import os
import shutil



shutil.rmtree("rsearch", ignore_errors=True)
os.mkdir("rsearch")
shutil.copyfile("python_file/__init__.py", "rsearch/__init__.py")
shutil.copyfile("rsearch.py", "rsearch/rsearch.py")
shutil.copyfile("python_file/handle_rsearch.py", "rsearch/handle_rsearch.py")
shutil.copyfile("_swigrsearch.so", "rsearch/_rsearch.so")
shutil.copyfile("../build/librsearch.so", "librsearch.so")
shutil.copyfile("../thirdparty/faiss/lib/libfaiss.so", "libfaiss.so")

setup(
    name="rsearch",
    version="0.1.0",
    description = "...",
    install_requires=['numpy'],
    packages=['rsearch'],
    package_data={
        'rsearch':['*.so'],
    },
)
