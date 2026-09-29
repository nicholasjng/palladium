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
        t9 = t0 * t6;
        float t10;
        t10 = t1 * t6;
        float t11;
        t11 = t10 * t7;
        float t12;
        t12 = t9 - t11;
        float t13;
        t13 = t2 * t6;
        float t14;
        t14 = t13 * t7;
        float t15;
        t15 = t3 * t7;
        float t16;
        t16 = t14 - t15;
        float t17;
        t17 = 0.004999999888241291f * t12;
        float t18;
        t18 = t6 + t17;
        float t19;
        t19 = 0.004999999888241291f * t16;
        float t20;
        t20 = t7 + t19;
        float t21;
        t21 = t0 * t18;
        float t22;
        t22 = t1 * t18;
        float t23;
        t23 = t22 * t20;
        float t24;
        t24 = t21 - t23;
        float t25;
        t25 = t2 * t18;
        float t26;
        t26 = t25 * t20;
        float t27;
        t27 = t3 * t20;
        float t28;
        t28 = t26 - t27;
        float t29;
        t29 = 0.004999999888241291f * t24;
        float t30;
        t30 = t6 + t29;
        float t31;
        t31 = 0.004999999888241291f * t28;
        float t32;
        t32 = t7 + t31;
        float t33;
        t33 = t0 * t30;
        float t34;
        t34 = t1 * t30;
        float t35;
        t35 = t34 * t32;
        float t36;
        t36 = t33 - t35;
        float t37;
        t37 = t2 * t30;
        float t38;
        t38 = t37 * t32;
        float t39;
        t39 = t3 * t32;
        float t40;
        t40 = t38 - t39;
        float t41;
        t41 = 0.009999999776482582f * t36;
        float t42;
        t42 = t6 + t41;
        float t43;
        t43 = 0.009999999776482582f * t40;
        float t44;
        t44 = t7 + t43;
        float t45;
        t45 = t0 * t42;
        float t46;
        t46 = t1 * t42;
        float t47;
        t47 = t46 * t44;
        float t48;
        t48 = t45 - t47;
        float t49;
        t49 = t2 * t42;
        float t50;
        t50 = t49 * t44;
        float t51;
        t51 = t3 * t44;
        float t52;
        t52 = t50 - t51;
        float t53;
        t53 = 2.0f * t24;
        float t54;
        t54 = t12 + t53;
        float t55;
        t55 = 2.0f * t36;
        float t56;
        t56 = t54 + t55;
        float t57;
        t57 = t56 + t48;
        float t58;
        t58 = 0.0016666667070239782f * t57;
        float t59;
        t59 = t6 + t58;
        float t60;
        t60 = 2.0f * t28;
        float t61;
        t61 = t16 + t60;
        float t62;
        t62 = 2.0f * t40;
        float t63;
        t63 = t61 + t62;
        float t64;
        t64 = t63 + t52;
        float t65;
        t65 = 0.0016666667070239782f * t64;
        float t66;
        t66 = t7 + t65;
        float _cb67 = t59;
        float _cb68 = t66;
        t6 = _cb67;
        t7 = _cb68;
    }
    arg6_offset[0] = t6;
    arg7_offset[0] = t7;
}
