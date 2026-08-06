#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0_base [[buffer(0)]],
    const device float* arg1_base [[buffer(1)]],
    device float* arg2_base [[buffer(2)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0 = arg0_base + (int)_pid.x * 8;
    const device float* arg1 = arg1_base + (int)_pid.x * 8;
    device float* arg2 = arg2_base + (int)_pid.x * 8;
    float t0[8];
    for (uint _i1 = 0; _i1 < 8; ++_i1) {
        t0[_i1] = arg0[_i1];
    }
    float t2[8];
    for (uint _i3 = 0; _i3 < 8; ++_i3) {
        t2[_i3] = (2.5f * t0[_i3]);
    }
    float t4[8];
    for (uint _i5 = 0; _i5 < 8; ++_i5) {
        t4[_i5] = arg1[_i5];
    }
    float t6[8];
    for (uint _i7 = 0; _i7 < 8; ++_i7) {
        t6[_i7] = (t2[_i7] + t4[_i7]);
    }
    for (uint _i8 = 0; _i8 < 8; ++_i8) {
        arg2[_i8] = t6[_i8];
    }
}
