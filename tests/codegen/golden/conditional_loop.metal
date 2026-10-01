#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0 [[buffer(0)]],
    const device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 _pid [[thread_position_in_grid]])
{
    float t0[64];
    for (uint _i1 = 0; _i1 < 64; ++_i1) {
        t0[_i1] = arg1[_i1];
    }
    float t2[64];
    for (uint _i3 = 0; _i3 < 64; ++_i3) {
        t2[_i3] = arg0[_i3];
    }
    for (uint _s4 = 0; _s4 < 20; ++_s4) {
        float t5[64];
        for (uint _i6 = 0; _i6 < 64; ++_i6) {
            t5[_i6] = t2[_i6] + t0[_i6];
        }
        float t7[64];
        for (uint _i8 = 0; _i8 < 64; ++_i8) {
            t7[_i8] = (t5[_i8] <= 1.0f) ? t5[_i8] : t2[_i8];
        }
        for (uint _i9 = 0; _i9 < 64; ++_i9) {
            float _cb10 = t7[_i9];
            t2[_i9] = _cb10;
        }
    }
    for (uint _i11 = 0; _i11 < 64; ++_i11) {
        arg2[_i11] = t2[_i11];
    }
}
