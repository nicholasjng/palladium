#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0_base [[buffer(0)]],
    device float* arg1_base [[buffer(1)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0 = arg0_base + 0;
    device float* arg1 = arg1_base + 0;
    float t0[512];
    for (uint _i1 = 0; _i1 < 512; ++_i1) {
        t0[_i1] = arg0[_i1];
    }
    for (uint _i2 = 0; _i2 < 512; ++_i2) {
        arg1[_i2] = t0[_i2];
    }
}
