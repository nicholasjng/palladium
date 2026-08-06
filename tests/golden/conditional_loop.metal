#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0_base [[buffer(0)]],
    const device float* arg1_base [[buffer(1)]],
    device float* arg2_base [[buffer(2)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0 = arg0_base + 0;
    const device float* arg1 = arg1_base + 0;
    device float* arg2 = arg2_base + 0;
    float t0[64];
    for (uint _i1 = 0; _i1 < 64; ++_i1) {
        t0[_i1] = arg1[_i1];
    }
    float t2[64];
    for (uint _i3 = 0; _i3 < 64; ++_i3) {
        t2[_i3] = arg0[_i3];
    }
    float t4[64];
    for (uint _i5 = 0; _i5 < 64; ++_i5) {
        t4[_i5] = t2[_i5];
    }
    for (uint _s6 = 0; _s6 < 20; ++_s6) {
        float t7[64];
        for (uint _i8 = 0; _i8 < 64; ++_i8) {
            t7[_i8] = t4[_i8] + t0[_i8];
        }
        bool t9[64];
        for (uint _i10 = 0; _i10 < 64; ++_i10) {
            t9[_i10] = t7[_i10] <= 1.0f;
        }
        float t11[64];
        for (uint _i12 = 0; _i12 < 64; ++_i12) {
            t11[_i12] = t9[_i12] ? t7[_i12] : t4[_i12];
        }
        for (uint _i13 = 0; _i13 < 64; ++_i13) {
            t4[_i13] = t11[_i13];
        }
    }
    for (uint _i14 = 0; _i14 < 64; ++_i14) {
        arg2[_i14] = t4[_i14];
    }
}
