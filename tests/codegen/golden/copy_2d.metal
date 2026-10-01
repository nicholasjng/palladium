#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0 [[buffer(0)]],
    device float* arg1 [[buffer(1)]],
    uint3 _pid [[thread_position_in_grid]])
{
    for (uint _i0 = 0; _i0 < 512; ++_i0) {
        arg1[_i0] = arg0[_i0];
    }
}
