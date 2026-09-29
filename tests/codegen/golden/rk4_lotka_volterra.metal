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
    const device float* arg0_offset = arg0 + (int)_pid.x * 1;
    const device float* arg1_offset = arg1 + (int)_pid.x * 1;
    const device float* arg2_offset = arg2 + (int)_pid.x * 1;
    const device float* arg3_offset = arg3 + (int)_pid.x * 1;
    const device float* arg4_offset = arg4 + (int)_pid.x * 1;
    const device float* arg5_offset = arg5 + (int)_pid.x * 1;
    device float* arg6_offset = arg6 + (int)_pid.x * 1;
    device float* arg7_offset = arg7 + (int)_pid.x * 1;
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
    float t6;
    t6 = t4;
    float t7;
    t7 = t5;
    for (uint _s8 = 0; _s8 < 500; ++_s8) {
        float t9;
        t9 = ((t0 * t6)) - ((((t1 * t6)) * t7));
        float t10;
        t10 = ((((t2 * t6)) * t7)) - ((t3 * t7));
        float t11;
        t11 = t6 + ((0.004999999888241291f * t9));
        float t12;
        t12 = t7 + ((0.004999999888241291f * t10));
        float t13;
        t13 = ((t0 * t11)) - ((((t1 * t11)) * t12));
        float t14;
        t14 = ((((t2 * t11)) * t12)) - ((t3 * t12));
        float t15;
        t15 = t6 + ((0.004999999888241291f * t13));
        float t16;
        t16 = t7 + ((0.004999999888241291f * t14));
        float t17;
        t17 = ((t0 * t15)) - ((((t1 * t15)) * t16));
        float t18;
        t18 = ((((t2 * t15)) * t16)) - ((t3 * t16));
        float t19;
        t19 = t6 + ((0.009999999776482582f * t17));
        float t20;
        t20 = t7 + ((0.009999999776482582f * t18));
        float t21;
        t21 = t6 + ((0.0016666667070239782f * ((((((t9 + ((2.0f * t13)))) + ((2.0f * t17)))) + ((((t0 * t19)) - ((((t1 * t19)) * t20))))))));
        float t22;
        t22 = t7 + ((0.0016666667070239782f * ((((((t10 + ((2.0f * t14)))) + ((2.0f * t18)))) + ((((((t2 * t19)) * t20)) - ((t3 * t20))))))));
        float _cb23 = t21;
        float _cb24 = t22;
        t6 = _cb23;
        t7 = _cb24;
    }
    arg6_offset[0] = t6;
    arg7_offset[0] = t7;
}
