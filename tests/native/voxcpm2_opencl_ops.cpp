#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

static void check(ggml_backend_t gpu, int kind, int k=1, int stride=1, int tin=7, int cin=3, int cout=5) {
    ggml_context *ctx = ggml_init({16*1024*1024, nullptr, true});
    ggml_tensor *out, *x=nullptr, *w=nullptr;
    std::vector<float> input, weights, expected;
    if (kind == 0 || kind >= 4) {
        x=ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 257);
        switch(kind) {
            case 0: out=ggml_sin(ctx,x); break;
            case 4: out=ggml_cos(ctx,x); break;
            case 5: out=ggml_sum(ctx,x); break;
            case 6: out=ggml_abs(ctx,x); break;
            case 7: out=ggml_step(ctx,x); break;
            case 8: out=ggml_round(ctx,x); break;
            default: throw std::runtime_error("Unknown test");
        }
        double sum=0;
        for(int i=0;i<257;i++) {
            float v=(i-123)*0.125f; input.push_back(v);sum+=v;
            if(kind==0)expected.push_back(std::sin(v));
            if(kind==4)expected.push_back(std::cos(v));
            if(kind==6)expected.push_back(std::abs(v));
            if(kind==7)expected.push_back(v>0 ? 1.0f : 0.0f);
            if(kind==8)expected.push_back(std::round(v));
        }
        if(kind==5)expected.push_back(float(sum));
    } else if (kind == 1) {
        out=ggml_arange(ctx, -0.75f, 1.5f, 0.125f);
        for(int i=0;i<ggml_nelements(out);i++) expected.push_back(-0.75f+0.125f*i);
    } else if (kind == 3) {
        out=ggml_arange(ctx, 1e-9f, 1.0f, 1.0f);
        expected.push_back(1e-9f);
    } else {
        x=ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tin, cin);
        w=ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, cout, cin);
        out=ggml_conv_transpose_1d(ctx,w,x,stride,0,1);
        for(int i=0;i<tin*cin;i++) input.push_back(std::sin(float(i)*0.731f));
        for(int i=0;i<k*cout*cin;i++) weights.push_back(std::cos(float(i)*0.193f)*0.1f);
        int tout=out->ne[0];
        expected.resize(ggml_nelements(out),0.0f);
        for(int co=0;co<cout;co++) for(int ti=0;ti<tin;ti++) for(int ki=0;ki<k;ki++) {
            double dot=0;
            for(int ci=0;ci<cin;ci++) dot+=double(input[ti+tin*ci])*weights[ki+k*(co+cout*ci)];
            expected[ti*stride+ki+tout*co]+=float(dot);
        }
    }
    if(!ggml_backend_supports_op(gpu,out)) throw std::runtime_error("GPU reports unsupported op");
    auto *graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);
    if(!buffer) throw std::runtime_error("Allocation failed");
    if(x)ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    if(w)ggml_backend_tensor_set(w,weights.data(),0,weights.size()*sizeof(float));
    if(ggml_backend_graph_compute(gpu,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU graph failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
    float max_error=0;
    for(size_t i=0;i<actual.size();i++) {
        if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite output");
        float error=std::abs(actual[i]-expected[i]);
        max_error=std::max(max_error,error);
        if(error>2e-5f+std::abs(expected[i])*1e-4f) {
            std::fprintf(stderr,"FAIL kind=%d at %zu: actual=%g expected=%g\n",kind,i,actual[i],expected[i]);
            throw std::runtime_error("Numerical mismatch");
        }
    }
    std::printf("PASS %s k=%d stride=%d tin=%d cin=%d cout=%d elements=%zu max_abs_error=%g\n",
                ggml_op_name(out->op),k,stride,tin,cin,cout,actual.size(),max_error);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
static void check_cast(ggml_backend_t gpu) {
    auto *ctx=ggml_init({1024*1024,nullptr,true});
    auto *x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,7);
    auto *out=ggml_cast(ctx,x,GGML_TYPE_I32);
    const float input[]={-3.9f,-1.0f,-0.5f,0,0.9f,1.1f,255.99f};
    const int32_t expected[]={-3,-1,0,0,0,1,255};
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    if(!ggml_backend_supports_op(gpu,out))throw std::runtime_error("Unsupported cast");
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);
    ggml_backend_tensor_set(x,input,0,sizeof(input));
    if(ggml_backend_graph_compute(gpu,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Cast compute failed");
    int32_t actual[7];ggml_backend_tensor_get(out,actual,0,sizeof(actual));
    for(int i=0;i<7;i++)if(actual[i]!=expected[i])throw std::runtime_error("Cast mismatch");
    std::puts("PASS CPY F32->I32 7 values including negatives and fractional inputs");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main() {
    ggml_backend_load_all();
    auto gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)throw std::runtime_error("No GPU backend");
    std::printf("Testing backend=%s\n",ggml_backend_name(gpu));
    check_cast(gpu);check(gpu,0);check(gpu,1);check(gpu,3);
    for(int kind=4;kind<=8;kind++)check(gpu,kind);
    for(int stride : {1,2,4,8})for(int k : {1,3,8,16})check(gpu,2,k,stride);
    check(gpu,2,8,4,17,128,64);
    ggml_backend_free(gpu);
}
