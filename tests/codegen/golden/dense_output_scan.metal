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
    float t2[4];
    for (uint _i3 = 0; _i3 < 4; ++_i3) {
        t2[_i3] = t0[_i3];
    }
    for (uint _s4 = 0; _s4 < 16; ++_s4) {
        float t5;
        t5 = 0.10000000149011612f * arg1[_s4];
        float t6[4];
        for (uint _i7 = 0; _i7 < 4; ++_i7) {
            t6[_i7] = t5 * t2[_i7];
        }
        float t8[4];
        for (uint _i9 = 0; _i9 < 4; ++_i9) {
            t8[_i9] = t2[_i9] + t6[_i9];
        }
        for (uint _i10 = 0; _i10 < 4; ++_i10) {
            (arg2 + _s4 * 4)[_i10] = t8[_i10];
        }
        for (uint _i11 = 0; _i11 < 4; ++_i11) {
            float _cb12 = t8[_i11];
            t2[_i11] = _cb12;
        }
    }
}
