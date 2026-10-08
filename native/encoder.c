#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include "blake2/blake2b-ref.c"

#define CACHE_SIZE 32768
#define CACHE_BYTES 20
/* A bounded direct-mapped cache of exact fragment digests, never predictions.
 * Length and bytes are compared on every hit; index collisions only evict.
 * CPython holds the GIL throughout this extension, including cache updates. */
typedef struct {
    uint64_t digest;
    unsigned char length;
    unsigned char bytes[CACHE_BYTES];
} CacheEntry;
static CacheEntry fragment_cache[CACHE_SIZE];

static int fragment_digest(uint64_t *value, const char *bytes, size_t count) {
    CacheEntry *entry = NULL;
    if (count > 0 && count <= CACHE_BYTES) {
        uint32_t index = 2166136261U;
        for (size_t i = 0; i < count; ++i)
            index = (index ^ (unsigned char)bytes[i]) * 16777619U;
        entry = &fragment_cache[index & (CACHE_SIZE - 1)];
        if (entry->length == count && memcmp(entry->bytes, bytes, count) == 0) {
            *value = entry->digest;
            return 0;
        }
    }
    unsigned char digest[8];
    if (blake2b(digest, sizeof(digest), bytes, count, NULL, 0) != 0)
        return -1;
    *value = 0;
    for (unsigned int b = 0; b < 8; ++b)
        *value |= ((uint64_t)digest[b]) << (8 * b);
    if (entry) {
        entry->digest = *value;
        entry->length = (unsigned char)count;
        memcpy(entry->bytes, bytes, count);
    }
    return 0;
}

static PyObject *cache_clear(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    memset(fragment_cache, 0, sizeof(fragment_cache));
    Py_RETURN_NONE;
}

/* Input is already normalized and padded by the Python encoder. Unicode
 * boundaries, not UTF-8 bytes, determine n-grams. The trained hash is unchanged. */
static PyObject *char_features(PyObject *self, PyObject *args) {
    PyObject *text, *grams, *bucket_arg, *sequence = NULL, *result = NULL;
    Py_ssize_t *boundaries = NULL;
    unsigned long long buckets;
    Py_ssize_t byte_length;
    (void)self;
    if (!PyArg_ParseTuple(args, "UOO:char_features", &text, &grams, &bucket_arg))
        return NULL;
    buckets = PyLong_AsUnsignedLongLong(bucket_arg);
    if (PyErr_Occurred()) return NULL;
    if (!buckets) {
        PyErr_SetString(PyExc_ValueError, "buckets must be positive");
        return NULL;
    }
    const char *utf8 = PyUnicode_AsUTF8AndSize(text, &byte_length);
    if (!utf8) return NULL;
    const Py_ssize_t length = PyUnicode_GetLength(text);
    if ((size_t)length > SIZE_MAX / sizeof(Py_ssize_t) - 1)
        return PyErr_NoMemory();
    boundaries = PyMem_Malloc(((size_t)length + 1) * sizeof(Py_ssize_t));
    if (!boundaries) return PyErr_NoMemory();
    Py_ssize_t character = 0;
    for (Py_ssize_t pos = 0; pos < byte_length; ++pos) {
        if (((unsigned char)utf8[pos] & 0xC0) != 0x80)
            boundaries[character++] = pos;
    }
    boundaries[length] = byte_length;
    sequence = PySequence_Fast(grams, "ngrams must be a sequence");
    if (!sequence) goto done;
    result = PyList_New(0);
    if (!result) goto done;
    for (Py_ssize_t g = 0; g < PySequence_Fast_GET_SIZE(sequence); ++g) {
        const Py_ssize_t n = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(sequence, g));
        if (n == -1 && PyErr_Occurred()) goto error;
        if (n < 1) {
            PyErr_SetString(PyExc_ValueError, "n-grams must be positive");
            goto error;
        }
        if (n > length) continue;
        for (Py_ssize_t i = 0; i <= length - n; ++i) {
            const Py_ssize_t start = boundaries[i];
            const size_t count = (size_t)(boundaries[i + n] - start);
            uint64_t value;
            if (fragment_digest(&value, utf8 + start, count) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "BLAKE2 hashing failed");
                goto error;
            }
            PyObject *index = PyLong_FromUnsignedLongLong(value % buckets);
            if (!index) goto error;
            const int status = PyList_Append(result, index);
            Py_DECREF(index);
            if (status < 0) goto error;
        }
    }
    goto done;
error:
    Py_CLEAR(result);
done:
    Py_XDECREF(sequence);
    PyMem_Free(boundaries);
    return result;
}

static PyMethodDef methods[] = {
    {"cache_clear", cache_clear, METH_NOARGS, "Clear the bounded fragment-digest cache."},
    {"char_features", char_features, METH_VARARGS,
     "Exact Unicode BLAKE2b-64 n-gram hashes for the Wind-Up checkpoint."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_windup_native", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__windup_native(void) { return PyModule_Create(&module); }
