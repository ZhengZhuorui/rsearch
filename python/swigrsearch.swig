%module rsearch
%{
#include "rsearch_type.h"
#include "probe/rsearch_probe.h"
#include "gallery/rsearch_gallery.h"

%}

%include "rsearch_type.h"
%include "probe/rsearch_probe.h"
%include "gallery/rsearch_gallery.h"
%include "rsearch_def.h"

namespace rsearch{

%template(create_probe_float) create_probe<float>;
%template(create_probe_int8) create_probe<int8_t>;
%template(probe_float) probe<float>;
%template(probe_int8) probe<int8_t>;
%template(gallery_float) gallery<float>;
%template(gallery_int8) gallery<int8_t>;
}