%module rsearch

%include <stdint.i>

%{
#include <bits/stdc++.h>
#include "rsearch_type.h"
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"
#include "rsearch.h"
#include "other/other.h"
#include "other/simple_index.h"

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
%}
%include<stdint.i>

namespace std {
    template<class T>
    class vector {
    public:
        vector();
        void push_back(T);
        void clear();
        T * data();
        size_t size();
        T at (size_t n) const;
        void resize (size_t n);
        void swap (vector<T> & other);
    };
};
%template(FloatVector) std::vector<float>;
%template(IntVector) std::vector<int>;
%template(Int8Vector) std::vector<int8_t>;
%template(QueryFormVector) std::vector<rsearch::query_form>;
%template(AreaTimeVector) std::vector<rsearch::area_time>;

%include "rsearch_type.h"
%include "other/other.h"
%include "other/simple_index.h"

%template(simple_index_areatime) rsearch::simple_index<rsearch::area_time>;
%include "rsearch.h"

void *memcpy(void *dest, const void *src, size_t n);



%{
PyObject *swig_ptr (PyObject *a)
{
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    /*if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }*/
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long_long, 0);
#endif
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long_long, 0);
#endif
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}
%}

// %include "numpy.i"
%init %{
    import_array();
%}
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)
%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);
