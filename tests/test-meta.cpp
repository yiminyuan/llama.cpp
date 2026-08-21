#include "ggml-backend.h"
#include "ggml.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// regression test: the scheduler skips view ops when forming split boundaries, so a view of a
// tensor outside the meta group (e.g. a state tensor owned by another simple group of a
// tensor-parallel-group setup, or a host tensor) can end up in a meta split; the meta backend
// must pass such no-op views through to the simple backends instead of requiring them to be
// allocated in a meta buffer

// all tensors of the meta group are mirrored: the test targets the placement of the view node
// in the splits, not the sharding of the data itself
static ggml_backend_meta_split_state test_meta_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    ggml_backend_meta_split_state ret{};
    ret.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    ret.nr[0] = 1;
    ret.n_segments = 1;
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return ret;
}

static void fill_tensor(ggml_tensor * t, float base) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; i++) {
        data[i] = 0.25f + 0.125f*float(i % 7) + base;
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
}

static ggml_context * new_ctx() {
    ggml_init_params params = { /*.mem_size =*/ 8*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    return ggml_init(params);
}

//
// a fake backend whose buffer type refuses to allocate above a configured size, used to force a
// deterministic out-of-memory in the meta backend allocation path without depending on the host
// having a particular amount of free memory
//

static ggml_status fake_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) { return GGML_STATUS_SUCCESS; }
static void fake_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}
static void fake_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, (const char *) data + offset, size);
}
static void fake_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy((char *) data + offset, (const char *) tensor->data + offset, size);
}
static void fake_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}
static void * fake_backend_buffer_get_base(ggml_backend_buffer_t buffer) { return buffer->context; }
static void   fake_backend_buffer_free(ggml_backend_buffer_t buffer) { std::free(buffer->context); }

struct fake_ctx {
    size_t max_buffer_size;
    bool   is_host;
};

static ggml_backend_buffer_t fake_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const fake_ctx * ctx = (fake_ctx *) buft->context;
    if (size > ctx->max_buffer_size) {
        return nullptr; // simulate out of memory
    }
    void * data = std::malloc(size ? size : 1);
    if (data == nullptr) {
        return nullptr;
    }
    static const struct ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ fake_backend_buffer_free,
        /* .get_base      = */ fake_backend_buffer_get_base,
        /* .init_tensor   = */ fake_backend_buffer_init_tensor,
        /* .memset_tensor = */ fake_backend_buffer_memset_tensor,
        /* .set_tensor    = */ fake_backend_buffer_set_tensor,
        /* .get_tensor    = */ fake_backend_buffer_get_tensor,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ fake_backend_buffer_clear,
        /* .reset         = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, data, size);
}

static const char * fake_backend_buft_get_name(ggml_backend_buffer_type_t) { return "fake"; }
static size_t fake_backend_buft_get_alignment(ggml_backend_buffer_type_t) { return 8; }
static bool   fake_backend_buft_is_host(ggml_backend_buffer_type_t buft) { return ((fake_ctx *) buft->context)->is_host; }

static ggml_backend_buffer_type_t fake_backend_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return (ggml_backend_buffer_type_t) dev->context;
}
static const char * fake_backend_dev_get_name(ggml_backend_dev_t) { return "fake"; }
static const char * fake_backend_dev_get_description(ggml_backend_dev_t) { return "fake device"; }
static void fake_backend_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = 0; *total = 0; }
static void fake_backend_dev_get_props(ggml_backend_dev_t, struct ggml_backend_dev_props * props) { std::memset(props, 0, sizeof(*props)); }
// the scheduler requires the last backend to be of CPU type; the non-host property the tests
// need comes from the buffer type, not the device
static enum ggml_backend_dev_type fake_backend_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; }
static bool fake_backend_dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool fake_backend_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t) { return true; }

// compute is delegated to a shared CPU backend; the fake device exists to provide a
// non-host buffer type, which is needed to trigger code paths that check
// ggml_backend_buffer_is_host
static ggml_guid_t fake_backend_guid(void) {
    static ggml_guid guid = { 0xfa, 0x6b, 0x91, 0x2c, 0x3d, 0x7e, 0x04, 0x8b, 0x61, 0xd2, 0x4f, 0xe3, 0x90, 0x5a, 0x27, 0x81 };
    return &guid;
}

static enum ggml_status fake_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return ggml_backend_graph_compute((ggml_backend_t) backend->context, cgraph);
}
static void fake_backend_free(ggml_backend_t backend) { delete (ggml_backend *) backend; }
static const char * fake_backend_name(ggml_backend_t) { return "fake"; }
static ggml_backend_buffer_type_t fake_backend_get_buffer_type(ggml_backend_t backend) {
    return fake_backend_dev_get_buffer_type(backend->device);
}

static const struct ggml_backend_i fake_backend_i = {
    /* .get_name           = */ fake_backend_name,
    /* .free               = */ fake_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ fake_backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static ggml_backend_t fake_backend_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    static ggml_backend_t cpu = nullptr;
    if (cpu == nullptr) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    }
    GGML_UNUSED(params);
    return new ggml_backend {
        /* .guid    = */ fake_backend_guid(),
        /* .iface   = */ fake_backend_i,
        /* .device  = */ dev,
        /* .context = */ cpu,
    };
}

// the objects are kept in static storage because ggml_backend_meta_device caches its input devices
static ggml_backend_dev_t fake_backend_create(size_t max_buffer_size, bool is_host = true) {
    static std::vector<std::unique_ptr<fake_ctx>>                          ctxs;
    static std::vector<std::unique_ptr<struct ggml_backend_buffer_type>> bufts;
    static std::vector<std::unique_ptr<struct ggml_backend_device>>     devs;

    auto ctx = std::make_unique<fake_ctx>();
    ctx->max_buffer_size = max_buffer_size;
    ctx->is_host         = is_host;

    auto buft = std::make_unique<struct ggml_backend_buffer_type>();
    buft->iface.get_name       = fake_backend_buft_get_name;
    buft->iface.alloc_buffer   = fake_backend_buft_alloc_buffer;
    buft->iface.get_alignment  = fake_backend_buft_get_alignment;
    buft->iface.get_max_size   = nullptr;
    buft->iface.get_alloc_size = nullptr;
    buft->iface.is_host        = fake_backend_buft_is_host;
    buft->device               = nullptr;
    buft->context              = ctx.get();

    auto dev = std::make_unique<struct ggml_backend_device>();
    dev->iface.get_name             = fake_backend_dev_get_name;
    dev->iface.get_description      = fake_backend_dev_get_description;
    dev->iface.get_memory           = fake_backend_dev_get_memory;
    dev->iface.get_type             = fake_backend_dev_get_type;
    dev->iface.get_props            = fake_backend_dev_get_props;
    dev->iface.init_backend         = fake_backend_dev_init_backend;
    dev->iface.get_buffer_type      = fake_backend_dev_get_buffer_type;
    dev->iface.get_host_buffer_type = nullptr;
    dev->iface.buffer_from_host_ptr = nullptr;
    dev->iface.supports_op          = fake_backend_dev_supports_op;
    dev->iface.supports_buft        = fake_backend_dev_supports_buft;
    dev->iface.offload_op           = nullptr;
    dev->iface.event_new            = nullptr;
    dev->iface.event_free           = nullptr;
    dev->iface.event_synchronize    = nullptr;
    dev->reg      = nullptr;
    dev->context  = buft.get();

    ctxs.push_back(std::move(ctx));
    bufts.push_back(std::move(buft));
    devs.push_back(std::move(dev));
    return devs.back().get();
}

// regression test: when a simple buffer of a meta buffer fails to allocate (e.g. out of memory),
// the meta allocation must release the meta buffer (freeing the simple buffers that were already
// allocated) and return NULL so the caller can report a clean error, instead of aborting
static int test_meta_alloc_oom() {
    // device A can allocate, device B cannot; the meta allocation must fail cleanly
    ggml_backend_dev_t dev_a = fake_backend_create(/*max_buffer_size =*/ 1 << 30);
    ggml_backend_dev_t dev_b = fake_backend_create(/*max_buffer_size =*/ 0);

    ggml_backend_dev_t meta_devs[2] = { dev_a, dev_b };
    ggml_backend_dev_t meta_dev     = ggml_backend_meta_device(meta_devs, 2, test_meta_get_split_state, nullptr);
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    ggml_context * ctx = new_ctx();
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_name(t, "t");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, meta_buft);
    if (buf != nullptr) {
        fprintf(stderr, "fail: expected the meta allocation to fail, but it succeeded\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }
    // the original tensors must be restored to the unallocated state
    if (t->buffer != nullptr || t->data != nullptr) {
        fprintf(stderr, "fail: tensor not restored to the unallocated state after the OOM (buffer = %p, data = %p)\n",
                (void *) t->buffer, t->data);
        ggml_free(ctx);
        return 1;
    }
    ggml_free(ctx);
    return 0;
}

// w and a are split along axis 0 (each device gets half the rows), so their mul_mat is a
// PARTIAL node: each device computes a partial sum that the meta backend all-reduces
static ggml_backend_meta_split_state test_meta_get_split_state_k(const struct ggml_tensor * tensor, void * userdata) {
    ggml_backend_meta_split_state ret{};
    ret.nr[0] = 1;
    ret.n_segments = 1;
    ret.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (std::string(ggml_get_name(tensor)) == "w" || std::string(ggml_get_name(tensor)) == "a") {
        ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
        ret.ne[0] = 2;
        ret.ne[1] = 2;
    }
    GGML_UNUSED(userdata);
    return ret;
}

// regression test: the AllReduce-delay scan in the meta graph compute walks every node after
// a PARTIAL node and queries their split state; a stray view of a non-host tensor in that range
// must be skipped (with a host view_src an is_host check masked the problem, so the simple
// group must be a non-host fake device to trigger it)
static int test_meta_stray_view_after_partial() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        fprintf(stderr, "fail: no CPU device\n");
        return 1;
    }

    ggml_backend_dev_t meta_devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev     = ggml_backend_meta_device(meta_devs, 2, test_meta_get_split_state_k, nullptr);
    ggml_backend_dev_t fake_dev     = fake_backend_create(/*max_buffer_size =*/ 1 << 30, /*is_host =*/ false);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t fake_backend = ggml_backend_dev_init(fake_dev, nullptr);
    if (meta_backend == nullptr || fake_backend == nullptr) {
        fprintf(stderr, "fail: backend init\n");
        return 1;
    }

    ggml_backend_t backends[2] = { meta_backend, fake_backend };
    ggml_backend_buffer_type_t bufts[2] = {
        ggml_backend_dev_buffer_type(meta_dev),
        ggml_backend_dev_buffer_type(fake_dev),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, /*parallel=*/false, /*op_offload=*/false);
    if (sched == nullptr) {
        fprintf(stderr, "fail: sched init\n");
        return 1;
    }

    const int64_t n = 4;

    // tensors owned by the meta group
    ggml_context * ctx_meta = new_ctx();
    ggml_tensor * w = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, n, n);
    ggml_set_name(w, "w");
    ggml_tensor * a = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, n, n);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, n, n);
    ggml_set_name(b, "b");
    ggml_backend_buffer_t meta_buf = ggml_backend_alloc_ctx_tensors(ctx_meta, meta_backend);

    // tensors owned by the simple group
    ggml_context * ctx_fake = new_ctx();
    ggml_tensor * x  = ggml_new_tensor_2d(ctx_fake, GGML_TYPE_F32, n, n);
    ggml_set_name(x, "x");
    ggml_tensor * a2 = ggml_new_tensor_2d(ctx_fake, GGML_TYPE_F32, n, n);
    ggml_set_name(a2, "a2");
    ggml_backend_buffer_t fake_buf = ggml_backend_alloc_ctx_tensors(ctx_fake, fake_backend);

    fill_tensor(w, 0.0f);
    fill_tensor(a, 1.0f);
    fill_tensor(b, 4.0f);
    fill_tensor(x, 2.0f);
    fill_tensor(a2, 3.0f);

    // m is PARTIAL in the meta group; the view of x belongs to the simple group but ends up
    // in the meta split after m; c is computed by the simple group from the view; m2 consumes
    // m (a PARTIAL tensor is not directly readable, its all-reduced result is read through m2)
    ggml_context * ctx_ops = new_ctx();
    ggml_tensor * m = ggml_mul_mat(ctx_ops, w, a);
    ggml_set_name(m, "m");
    ggml_tensor * v = ggml_view_2d(ctx_ops, x, n, n, x->nb[1], 0);
    ggml_set_name(v, "v");
    ggml_tensor * c = ggml_mul_mat(ctx_ops, v, a2);
    ggml_set_name(c, "c");
    ggml_tensor * m2 = ggml_add(ctx_ops, m, b);
    ggml_set_name(m2, "m2");

    // build order matters: m, then v and c, then m2, so that the stray view v sits between the
    // PARTIAL node m and m2, which is the first node after m that consumes m
    ggml_cgraph * gf = ggml_new_graph(ctx_ops);
    ggml_build_forward_expand(gf, m);
    ggml_build_forward_expand(gf, c);
    ggml_build_forward_expand(gf, m2);

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fail: graph compute\n");
        return 1;
    }

    std::vector<float> w_d(n*n), a_d(n*n), b_d(n*n), x_d(n*n), a2_d(n*n);
    ggml_backend_tensor_get(w,  w_d.data(),  0, w_d.size()*sizeof(float));
    ggml_backend_tensor_get(a,  a_d.data(),  0, a_d.size()*sizeof(float));
    ggml_backend_tensor_get(b,  b_d.data(),  0, b_d.size()*sizeof(float));
    ggml_backend_tensor_get(x,  x_d.data(),  0, x_d.size()*sizeof(float));
    ggml_backend_tensor_get(a2, a2_d.data(), 0, a2_d.size()*sizeof(float));

    // ggml_mul_mat(ctx, A, B) computes C = B A^T: C[i][j] = sum_k B[i][k] * A[j][k]
    std::vector<float> m2_ref(n*n), c_ref(n*n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            float m_acc = 0.0f;
            float c_acc = 0.0f;
            for (int64_t k = 0; k < n; k++) {
                m_acc += a_d[i*n + k]  * w_d[j*n + k];
                c_acc += a2_d[i*n + k] * x_d[j*n + k];
            }
            m2_ref[i*n + j] = m_acc + b_d[i*n + j];
            c_ref[i*n + j]  = c_acc;
        }
    }

    std::vector<float> m2_out(n*n), c_out(n*n);
    ggml_backend_tensor_get(m2, m2_out.data(), 0, m2_out.size()*sizeof(float));
    ggml_backend_tensor_get(c,  c_out.data(),  0, c_out.size()*sizeof(float));

    bool ok = true;
    for (int64_t i = 0; i < n*n; i++) {
        if (std::fabs(m2_out[i] - m2_ref[i]) > 1e-4f) {
            fprintf(stderr, "fail: m2[%ld] = %f, expected %f\n", (long) i, m2_out[i], m2_ref[i]);
            ok = false;
        }
        if (std::fabs(c_out[i] - c_ref[i]) > 1e-4f) {
            fprintf(stderr, "fail: c[%ld] = %f, expected %f\n", (long) i, c_out[i], c_ref[i]);
            ok = false;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(meta_buf);
    ggml_backend_buffer_free(fake_buf);
    ggml_backend_free(meta_backend);
    ggml_backend_free(fake_backend);
    ggml_free(ctx_ops);
    ggml_free(ctx_meta);
    ggml_free(ctx_fake);

    return ok ? 0 : 1;
}

int main() {
    ggml_backend_load_all();

    // a simple buffer failing to allocate must make the meta allocation fail cleanly, not abort
    if (test_meta_alloc_oom() != 0) {
        return 1;
    }

    // a stray view after a PARTIAL node must not abort the AllReduce-delay scan
    if (test_meta_stray_view_after_partial() != 0) {
        return 1;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        fprintf(stderr, "fail: no CPU device\n");
        return 1;
    }

    // a 2-device meta group (both ranks on the CPU backend) plus the plain CPU backend as a
    // second, simple group
    ggml_backend_dev_t meta_devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(meta_devs, 2, test_meta_get_split_state, nullptr);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t cpu_backend  = ggml_backend_dev_init(cpu_dev, nullptr);
    if (meta_backend == nullptr || cpu_backend == nullptr) {
        fprintf(stderr, "fail: backend init\n");
        return 1;
    }

    ggml_backend_t backends[2] = { meta_backend, cpu_backend };
    ggml_backend_buffer_type_t bufts[2] = {
        ggml_backend_dev_buffer_type(meta_dev),
        ggml_backend_dev_buffer_type(cpu_dev),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, /*parallel=*/false, /*op_offload=*/false);
    if (sched == nullptr) {
        fprintf(stderr, "fail: sched init\n");
        return 1;
    }

    const int64_t n = 4;

    // tensors owned by the meta group
    ggml_context * ctx_meta = new_ctx();
    ggml_tensor * w = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, n, n);
    ggml_set_name(w, "w");
    ggml_tensor * a = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, n, n);
    ggml_set_name(a, "a");
    ggml_backend_buffer_t meta_buf = ggml_backend_alloc_ctx_tensors(ctx_meta, meta_backend);

    // tensors owned by the simple group
    ggml_context * ctx_cpu = new_ctx();
    ggml_tensor * x  = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, n, n);
    ggml_set_name(x, "x");
    ggml_tensor * a2 = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, n, n);
    ggml_set_name(a2, "a2");
    ggml_backend_buffer_t cpu_buf = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu_backend);

    fill_tensor(w, 0.0f);
    fill_tensor(a, 1.0f);
    fill_tensor(x, 2.0f);
    fill_tensor(a2, 3.0f);

    // m is computed by the meta group; the view of x belongs to the simple group, but ends up
    // in the meta split because the scheduler skips view ops when forming split boundaries; c
    // is computed by the simple group from the view
    ggml_context * ctx_ops = new_ctx();
    ggml_tensor * m = ggml_mul_mat(ctx_ops, w, a);
    ggml_set_name(m, "m");
    ggml_tensor * v = ggml_view_2d(ctx_ops, x, n, n, x->nb[1], 0);
    ggml_set_name(v, "v");
    ggml_tensor * c = ggml_mul_mat(ctx_ops, v, a2);
    ggml_set_name(c, "c");

    ggml_cgraph * gf = ggml_new_graph(ctx_ops);
    ggml_build_forward_expand(gf, m);
    ggml_build_forward_expand(gf, c);

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fail: graph compute\n");
        return 1;
    }

    std::vector<float> w_d(n*n), a_d(n*n), x_d(n*n), a2_d(n*n);
    ggml_backend_tensor_get(w,  w_d.data(),  0, w_d.size()*sizeof(float));
    ggml_backend_tensor_get(a,  a_d.data(),  0, a_d.size()*sizeof(float));
    ggml_backend_tensor_get(x,  x_d.data(),  0, x_d.size()*sizeof(float));
    ggml_backend_tensor_get(a2, a2_d.data(), 0, a2_d.size()*sizeof(float));

    // ggml_mul_mat(ctx, A, B) computes C = B A^T: C[i][j] = sum_k B[i][k] * A[j][k]
    std::vector<float> m_ref(n*n), c_ref(n*n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            float m_acc = 0.0f;
            float c_acc = 0.0f;
            for (int64_t k = 0; k < n; k++) {
                m_acc += a_d[i*n + k]  * w_d[j*n + k];
                c_acc += a2_d[i*n + k] * x_d[j*n + k];
            }
            m_ref[i*n + j] = m_acc;
            c_ref[i*n + j] = c_acc;
        }
    }

    std::vector<float> m_out(n*n), c_out(n*n);
    ggml_backend_tensor_get(m, m_out.data(), 0, m_out.size()*sizeof(float));
    ggml_backend_tensor_get(c, c_out.data(), 0, c_out.size()*sizeof(float));

    bool ok = true;
    for (int64_t i = 0; i < n*n; i++) {
        if (std::fabs(m_out[i] - m_ref[i]) > 1e-4f) {
            fprintf(stderr, "fail: m[%ld] = %f, expected %f\n", (long) i, m_out[i], m_ref[i]);
            ok = false;
        }
        if (std::fabs(c_out[i] - c_ref[i]) > 1e-4f) {
            fprintf(stderr, "fail: c[%ld] = %f, expected %f\n", (long) i, c_out[i], c_ref[i]);
            ok = false;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(meta_buf);
    ggml_backend_buffer_free(cpu_buf);
    ggml_backend_free(meta_backend);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx_ops);
    ggml_free(ctx_meta);
    ggml_free(ctx_cpu);

    if (!ok) {
        return 1;
    }

    printf("test-meta: PASSED\n");
    return 0;
}
