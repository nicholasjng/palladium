#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0 [[buffer(0)]],
    const device float* arg1 [[buffer(1)]],
    const device float* arg2 [[buffer(2)]],
    const device float* arg3 [[buffer(3)]],
    const device float* arg4 [[buffer(4)]],
    const device float* arg5 [[buffer(5)]],
    device float* arg6 [[buffer(6)]],
    device float* arg7 [[buffer(7)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0_offset = arg0 + (int)_pid.x;
    const device float* arg1_offset = arg1 + (int)_pid.x;
    const device float* arg2_offset = arg2 + (int)_pid.x;
    const device float* arg3_offset = arg3 + (int)_pid.x;
    const device float* arg4_offset = arg4 + (int)_pid.x;
    const device float* arg5_offset = arg5 + (int)_pid.x;
    device float* arg6_offset = arg6 + (int)_pid.x;
    device float* arg7_offset = arg7 + (int)_pid.x;
    float t0;
    t0 = arg2_offset[0];
    float t1;
    t1 = arg3_offset[0];
    float t2;
    t2 = arg4_offset[0];
    float t3;
    t3 = arg5_offset[0];
    float t4;
    t4 = arg0_offset[0];
    float t5;
    t5 = arg1_offset[0];
    for (uint _s6 = 0; _s6 < 500; ++_s6) {
        float t7;
        t7 = (t0 * t4) - ((t1 * t4) * t5);
        float t8;
        t8 = ((t2 * t4) * t5) - (t3 * t5);
        float t9;
        t9 = t4 + (0.005f * t7);
        float t10;
        t10 = t5 + (0.005f * t8);
        float t11;
        t11 = (t0 * t9) - ((t1 * t9) * t10);
        float t12;
        t12 = ((t2 * t9) * t10) - (t3 * t10);
        float t13;
        t13 = t4 + (0.005f * t11);
        float t14;
        t14 = t5 + (0.005f * t12);
        float t15;
        t15 = (t0 * t13) - ((t1 * t13) * t14);
        float t16;
        t16 = ((t2 * t13) * t14) - (t3 * t14);
        float t17;
        t17 = t4 + (0.01f * t15);
        float t18;
        t18 = t5 + (0.01f * t16);
        float t19;
        t19 = t4 + (0.0016666667f * (((t7 + (2.0f * t11)) + (2.0f * t15)) + ((t0 * t17) - ((t1 * t17) * t18))));
        float t20;
        t20 = t5 + (0.0016666667f * (((t8 + (2.0f * t12)) + (2.0f * t16)) + (((t2 * t17) * t18) - (t3 * t18))));
        float _cb21 = t19;
        float _cb22 = t20;
        t4 = _cb21;
        t5 = _cb22;
    }
    arg6_offset[0] = t4;
    arg7_offset[0] = t5;
}
