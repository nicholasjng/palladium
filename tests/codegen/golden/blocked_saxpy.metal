#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0 [[buffer(0)]],
    const device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0_offset = arg0 + (int)_pid.x * 8;
    const device float* arg1_offset = arg1 + (int)_pid.x * 8;
    device float* arg2_offset = arg2 + (int)_pid.x * 8;
    float t0[8];
    for (uint _i1 = 0; _i1 < 8; ++_i1) {
        t0[_i1] = arg0_offset[_i1];
    }
    float t2[8];
    for (uint _i3 = 0; _i3 < 8; ++_i3) {
        t2[_i3] = arg1_offset[_i3];
    }
    for (uint _i4 = 0; _i4 < 8; ++_i4) {
        arg2_offset[_i4] = (2.5f * t0[_i4]) + t2[_i4];
    }
}
