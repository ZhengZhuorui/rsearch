
-include ../makefile.inc

all: build

swigrsearch_wrap.cpp: swigrsearch.swig ../build/librsearch.so
	$(SWIG) -python -c++ -I../include -o $@ $<
swigrsearch_wrap.o: swigrsearch_wrap.cpp
	$(CXX) -c -fPIC swigrsearch_wrap.cpp -I/usr/include/python3.5 -I../include -I../thirdparty/faiss/include
_swigrsearch.so: swigrsearch_wrap.o ../build/librsearch.so
	$(CXX) -shared -o $@ $^
build: _swigrsearch.so
	$(PYTHON) setup.py build

clean:
	rm -f swigrsearch_wrap.cpp swigrsearch_wrap.o _swigrsearch.so
	rm -rf build

.PHONY: all build clean
