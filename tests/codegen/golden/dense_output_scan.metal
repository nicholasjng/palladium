#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0 [[buffer(0)]],
    const device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 _pid [[thread_position_in_grid]])
{
    float t0[4];
    for (uint _i1 = 0; _i1 < 4; ++_i1) {
        t0[_i1] = arg0[_i1];
    }
    for (uint _s2 = 0; _s2 < 16; ++_s2) {
        float t3;
        t3 = 0.1f * arg1[_s2];
        float t4[4];
        for (uint _i5 = 0; _i5 < 4; ++_i5) {
            t4[_i5] = t0[_i5] + (t3 * t0[_i5]);
        }
        for (uint _i6 = 0; _i6 < 4; ++_i6) {
            (arg2 + _s2 * 4)[_i6] = t4[_i6];
        }
        for (uint _i7 = 0; _i7 < 4; ++_i7) {
            float _cb8 = t4[_i7];
            t0[_i7] = _cb8;
        }
    }
}
