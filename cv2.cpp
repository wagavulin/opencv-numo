#include <ruby.h>
#include <numo/narray.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>

void dbprint(const char* filename, int line, const char* funcname, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%s %d %s] ", filename, line, funcname);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

#define DBPRINT(fmt, ...) dbprint(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

using namespace cv;

static VALUE mCV2;

const char* db_get_class_name(VALUE o){
    VALUE vtmp1 = rb_funcall(o, rb_intern("class"), 0, 0);
    VALUE vtmp2 = rb_funcall(vtmp1, rb_intern("to_s"), 0, 0);
    return StringValuePtr(vtmp2);
}

void db_dump_narray(int level, const narray_t* na){
    printf("%*sndim: %d, type: %d, flag: [%d,%d], elmsz: %d, size: %ld\n", level*2, "", na->ndim, na->type, na->flag[0], na->flag[1], na->elmsz, na->size);
    for (unsigned char i = 0; i < na->ndim; i++) {
        printf("%*sshape[%d]: %ld\n", (level+1)*2, "", i, na->shape[i]);
    }
    printf("%*sreduce: %s: %d\n", level*2, "", db_get_class_name(na->reduce), NUM2INT(na->reduce));
}

void db_dump_narray_data(int level, const narray_data_t* nad){
    db_dump_narray(level, &nad->base);
    printf("%*sowned: %d\n", level*2, "", nad->owned);
}

void db_dump_narray_view(int level, const narray_view_t* nav){
    int i;
    size_t *idx;
    size_t j;

    printf("  offset = %ld\n", (size_t)nav->offset);
    printf("  stridx = %ld\n", (size_t)nav->stridx);

    if (nav->stridx) {
        printf("  stridx = [");
        for (i=0; i<nav->base.ndim; i++) {
            if (SDX_IS_INDEX(nav->stridx[i])) {

                idx = SDX_GET_INDEX(nav->stridx[i]);
                printf("  index[%d]=[", i);
                for (j=0; j<nav->base.shape[i]; j++) {
                    printf(" %ld", idx[j]);
                }
                printf(" ] ");

            } else {
                printf(" %ld", SDX_GET_STRIDE(nav->stridx[i]));
            }
        }
        printf(" ]\n");
    }
}

template<typename T>
static bool rbopencv_to(VALUE obj, T& p){
    DBPRINT("\n");
    throw std::runtime_error("generic rbopencv_to");
}

class NumpyAllocator : public cv::MatAllocator {
public:
    NumpyAllocator() { stdAllocator = cv::Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    UMatData* allocate(VALUE o, int dims, const int* sizes, int type, size_t* step) const {
        narray_data_t* nad = na_get_narray_data_t(o);
        VALUE view = rb_funcall(o, rb_intern("view"), 0, 0);
        narray_view_t* nav = na_get_narray_view_t(view);

        UMatData* u = new UMatData(this);
        DBPRINT("  u: %p\n", u);
        u->data = u->origdata = (uchar*)nad->ptr;
        if (!nav->stridx) {
            throw std::runtime_error("[NumpyAllocator::allocate] nav->stridx is NULL");
        }
        for (unsigned char i = 0; i < nad->base.ndim; i++) {
            if (SDX_IS_INDEX(nav->stridx[i])) {
                DBPRINT("nav->stridx[%d] is not stride\n", i);
                throw std::runtime_error("[NumpyAllocator::allocate] nav->stridx[i] is not stride");
            } else {
                ssize_t stride = SDX_GET_STRIDE(nav->stridx[i]);
                step[i] = (size_t)stride;
                //printf("step[%d]: %ld\n", i, step[i]);
            }
        }
        step[dims-1] = CV_ELEM_SIZE(type);
        //printf("step[dims-1=%d]: %ld\n", dims-1, step[dims-1]);
        u->size = sizes[0] * step[0];
        //printf("u->size: %ld\n", u->size);
        u->userdata = (void*)o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const override {
        DBPRINT("dims0: %d, type: %d, depth: %d, cn: %d\n", dims0, type, CV_MAT_DEPTH(type), CV_MAT_CN(type));
        for (int i = 0; i < dims0; i++) {
            DBPRINT("  sizes[%d]: %d\n", i, sizes[i]);
        }
        if (data) {
            throw std::runtime_error("[NumpyAllocator::allocate] data is not NULL");
        }
        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        VALUE numo_type = depth == CV_8U ? numo_cUInt8 : depth == CV_8S ? numo_cInt8 :
        depth == CV_16U ? numo_cUInt16 : depth == CV_16S ? numo_cInt16 :
        depth == CV_32S ? numo_cInt32 : depth == CV_32F ? numo_cSFloat :
        depth == CV_64F ? numo_cDFloat : 0xffff;
        if (numo_type == 0xffff) {
            throw std::runtime_error("[NumpyAllocator::allocate] Unsupported type\n");
        }

        int i, dims = dims0;
        cv::AutoBuffer<size_t> _sizes(dims + 1);
        for (i = 0; i < dims; i++) {
            _sizes[i] = sizes[i];
        }
        if (cn > 1) {
            _sizes[dims++] = cn;
        }
        VALUE o = rb_narray_new(numo_type, dims, _sizes.data());
        rb_funcall(o, rb_intern("fill"), 1, INT2FIX(3));

        cv::UMatData* ret = allocate(o, dims0, sizes, type, step);
        DBPRINT("ret: %p\n", ret);
        return ret;
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const override {
        DBPRINT("\n");
        return false;
    }

    void deallocate(UMatData* u) const override {
        DBPRINT("%p\n", u);
        if (!u)
            return;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if (u->refcount == 0) {
            DBPRINT("  refcount == 0; delete %p\n", u);
            delete u;
        } else {
            DBPRINT("  refcount >= 1\n");
        }
    }

    const cv::MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;

static bool rbopencv_to(VALUE o, Mat& m){
    DBPRINT("o: %s\n", db_get_class_name(o));
    bool allowND = true;
    if (NIL_P(o)) {
        DBPRINT("  o is NIL\n");
        if (!m.data)
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if (TYPE(o) == T_FIXNUM) {
        DBPRINT("  o is FIXNUM\n");
        return false;
    }
    if (TYPE(o) == T_FLOAT) {
        DBPRINT("  o is \n");
        return false;
    }
    return false;
}

template<typename T>
static VALUE rbopencv_from(const T& src){
    throw std::runtime_error("Generic rbopencv_from is called");
}

template<>
VALUE rbopencv_from(const cv::Mat& m){
    if (!m.data) {
        DBPRINT("m.data is null\n");
        return Qnil;
    }
    DBPRINT("m.u: %p\n", m.u);
    cv::Mat temp, *p = (cv::Mat*)&m;
    if (!p->u || p->allocator != &g_numpyAllocator) {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    VALUE o = (VALUE)p->u->userdata;
    return o;
}

static VALUE wrap_imread(VALUE self, VALUE filename, VALUE flags){
    const char* raw_filename = RSTRING_PTR(filename);
    int raw_flags = NUM2INT(flags);
    cv::Mat m = cv::imread(raw_filename, raw_flags);
    VALUE ret = rbopencv_from(m);
    return ret;
}

static VALUE wrap_circle_test(VALUE self, VALUE rbobj_img){
    DBPRINT("\n");
    narray_data_t* nad = na_get_narray_data_t(rbobj_img);
    DBPRINT("  ptr: %p\n", nad->ptr);

    cv::Mat img;
    rbopencv_to(rbobj_img, img);
    return Qnil;
}

extern "C" {
void Init_cv2(){
    mCV2 = rb_define_module("CV2");

    rb_define_const(mCV2, "IMREAD_GRAYSCALE", INT2FIX(0));
    rb_define_const(mCV2, "IMREAD_COLOR", INT2FIX(1));

    rb_define_module_function(mCV2, "imread", RUBY_METHOD_FUNC(wrap_imread), 2);
    rb_define_module_function(mCV2, "circle_test", RUBY_METHOD_FUNC(wrap_circle_test), 1);
}
}
