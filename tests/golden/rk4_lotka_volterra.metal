#include <metal_stdlib>
using namespace metal;

kernel void palladium_kernel(
    const device float* arg0_base [[buffer(0)]],
    const device float* arg1_base [[buffer(1)]],
    const device float* arg2_base [[buffer(2)]],
    const device float* arg3_base [[buffer(3)]],
    const device float* arg4_base [[buffer(4)]],
    const device float* arg5_base [[buffer(5)]],
    device float* arg6_base [[buffer(6)]],
    device float* arg7_base [[buffer(7)]],
    uint3 _pid [[thread_position_in_grid]])
{
    const device float* arg0 = arg0_base + (int)_pid.x * 1;
    const device float* arg1 = arg1_base + (int)_pid.x * 1;
    const device float* arg2 = arg2_base + (int)_pid.x * 1;
    const device float* arg3 = arg3_base + (int)_pid.x * 1;
    const device float* arg4 = arg4_base + (int)_pid.x * 1;
    const device float* arg5 = arg5_base + (int)_pid.x * 1;
    device float* arg6 = arg6_base + (int)_pid.x * 1;
    device float* arg7 = arg7_base + (int)_pid.x * 1;
    float t0[1];
    for (uint _i1 = 0; _i1 < 1; ++_i1) {
        t0[_i1] = arg2[_i1];
    }
    float t2[1];
    for (uint _i3 = 0; _i3 < 1; ++_i3) {
        t2[_i3] = arg3[_i3];
    }
    float t4[1];
    for (uint _i5 = 0; _i5 < 1; ++_i5) {
        t4[_i5] = arg4[_i5];
    }
    float t6[1];
    for (uint _i7 = 0; _i7 < 1; ++_i7) {
        t6[_i7] = arg5[_i7];
    }
    float t8[1];
    for (uint _i9 = 0; _i9 < 1; ++_i9) {
        t8[_i9] = arg0[_i9];
    }
    float t10[1];
    for (uint _i11 = 0; _i11 < 1; ++_i11) {
        t10[_i11] = arg1[_i11];
    }
    float t12[1];
    for (uint _i13 = 0; _i13 < 1; ++_i13) {
        t12[_i13] = t8[_i13];
    }
    float t14[1];
    for (uint _i15 = 0; _i15 < 1; ++_i15) {
        t14[_i15] = t10[_i15];
    }
    for (uint _s16 = 0; _s16 < 500; ++_s16) {
        float t17[1];
        for (uint _i18 = 0; _i18 < 1; ++_i18) {
            t17[_i18] = t0[_i18] * t12[_i18];
        }
        float t19[1];
        for (uint _i20 = 0; _i20 < 1; ++_i20) {
            t19[_i20] = t2[_i20] * t12[_i20];
        }
        float t21[1];
        for (uint _i22 = 0; _i22 < 1; ++_i22) {
            t21[_i22] = t19[_i22] * t14[_i22];
        }
        float t23[1];
        for (uint _i24 = 0; _i24 < 1; ++_i24) {
            t23[_i24] = t17[_i24] - t21[_i24];
        }
        float t25[1];
        for (uint _i26 = 0; _i26 < 1; ++_i26) {
            t25[_i26] = t4[_i26] * t12[_i26];
        }
        float t27[1];
        for (uint _i28 = 0; _i28 < 1; ++_i28) {
            t27[_i28] = t25[_i28] * t14[_i28];
        }
        float t29[1];
        for (uint _i30 = 0; _i30 < 1; ++_i30) {
            t29[_i30] = t6[_i30] * t14[_i30];
        }
        float t31[1];
        for (uint _i32 = 0; _i32 < 1; ++_i32) {
            t31[_i32] = t27[_i32] - t29[_i32];
        }
        float t33[1];
        for (uint _i34 = 0; _i34 < 1; ++_i34) {
            t33[_i34] = 0.004999999888241291f * t23[_i34];
        }
        float t35[1];
        for (uint _i36 = 0; _i36 < 1; ++_i36) {
            t35[_i36] = t12[_i36] + t33[_i36];
        }
        float t37[1];
        for (uint _i38 = 0; _i38 < 1; ++_i38) {
            t37[_i38] = 0.004999999888241291f * t31[_i38];
        }
        float t39[1];
        for (uint _i40 = 0; _i40 < 1; ++_i40) {
            t39[_i40] = t14[_i40] + t37[_i40];
        }
        float t41[1];
        for (uint _i42 = 0; _i42 < 1; ++_i42) {
            t41[_i42] = t0[_i42] * t35[_i42];
        }
        float t43[1];
        for (uint _i44 = 0; _i44 < 1; ++_i44) {
            t43[_i44] = t2[_i44] * t35[_i44];
        }
        float t45[1];
        for (uint _i46 = 0; _i46 < 1; ++_i46) {
            t45[_i46] = t43[_i46] * t39[_i46];
        }
        float t47[1];
        for (uint _i48 = 0; _i48 < 1; ++_i48) {
            t47[_i48] = t41[_i48] - t45[_i48];
        }
        float t49[1];
        for (uint _i50 = 0; _i50 < 1; ++_i50) {
            t49[_i50] = t4[_i50] * t35[_i50];
        }
        float t51[1];
        for (uint _i52 = 0; _i52 < 1; ++_i52) {
            t51[_i52] = t49[_i52] * t39[_i52];
        }
        float t53[1];
        for (uint _i54 = 0; _i54 < 1; ++_i54) {
            t53[_i54] = t6[_i54] * t39[_i54];
        }
        float t55[1];
        for (uint _i56 = 0; _i56 < 1; ++_i56) {
            t55[_i56] = t51[_i56] - t53[_i56];
        }
        float t57[1];
        for (uint _i58 = 0; _i58 < 1; ++_i58) {
            t57[_i58] = 0.004999999888241291f * t47[_i58];
        }
        float t59[1];
        for (uint _i60 = 0; _i60 < 1; ++_i60) {
            t59[_i60] = t12[_i60] + t57[_i60];
        }
        float t61[1];
        for (uint _i62 = 0; _i62 < 1; ++_i62) {
            t61[_i62] = 0.004999999888241291f * t55[_i62];
        }
        float t63[1];
        for (uint _i64 = 0; _i64 < 1; ++_i64) {
            t63[_i64] = t14[_i64] + t61[_i64];
        }
        float t65[1];
        for (uint _i66 = 0; _i66 < 1; ++_i66) {
            t65[_i66] = t0[_i66] * t59[_i66];
        }
        float t67[1];
        for (uint _i68 = 0; _i68 < 1; ++_i68) {
            t67[_i68] = t2[_i68] * t59[_i68];
        }
        float t69[1];
        for (uint _i70 = 0; _i70 < 1; ++_i70) {
            t69[_i70] = t67[_i70] * t63[_i70];
        }
        float t71[1];
        for (uint _i72 = 0; _i72 < 1; ++_i72) {
            t71[_i72] = t65[_i72] - t69[_i72];
        }
        float t73[1];
        for (uint _i74 = 0; _i74 < 1; ++_i74) {
            t73[_i74] = t4[_i74] * t59[_i74];
        }
        float t75[1];
        for (uint _i76 = 0; _i76 < 1; ++_i76) {
            t75[_i76] = t73[_i76] * t63[_i76];
        }
        float t77[1];
        for (uint _i78 = 0; _i78 < 1; ++_i78) {
            t77[_i78] = t6[_i78] * t63[_i78];
        }
        float t79[1];
        for (uint _i80 = 0; _i80 < 1; ++_i80) {
            t79[_i80] = t75[_i80] - t77[_i80];
        }
        float t81[1];
        for (uint _i82 = 0; _i82 < 1; ++_i82) {
            t81[_i82] = 0.009999999776482582f * t71[_i82];
        }
        float t83[1];
        for (uint _i84 = 0; _i84 < 1; ++_i84) {
            t83[_i84] = t12[_i84] + t81[_i84];
        }
        float t85[1];
        for (uint _i86 = 0; _i86 < 1; ++_i86) {
            t85[_i86] = 0.009999999776482582f * t79[_i86];
        }
        float t87[1];
        for (uint _i88 = 0; _i88 < 1; ++_i88) {
            t87[_i88] = t14[_i88] + t85[_i88];
        }
        float t89[1];
        for (uint _i90 = 0; _i90 < 1; ++_i90) {
            t89[_i90] = t0[_i90] * t83[_i90];
        }
        float t91[1];
        for (uint _i92 = 0; _i92 < 1; ++_i92) {
            t91[_i92] = t2[_i92] * t83[_i92];
        }
        float t93[1];
        for (uint _i94 = 0; _i94 < 1; ++_i94) {
            t93[_i94] = t91[_i94] * t87[_i94];
        }
        float t95[1];
        for (uint _i96 = 0; _i96 < 1; ++_i96) {
            t95[_i96] = t89[_i96] - t93[_i96];
        }
        float t97[1];
        for (uint _i98 = 0; _i98 < 1; ++_i98) {
            t97[_i98] = t4[_i98] * t83[_i98];
        }
        float t99[1];
        for (uint _i100 = 0; _i100 < 1; ++_i100) {
            t99[_i100] = t97[_i100] * t87[_i100];
        }
        float t101[1];
        for (uint _i102 = 0; _i102 < 1; ++_i102) {
            t101[_i102] = t6[_i102] * t87[_i102];
        }
        float t103[1];
        for (uint _i104 = 0; _i104 < 1; ++_i104) {
            t103[_i104] = t99[_i104] - t101[_i104];
        }
        float t105[1];
        for (uint _i106 = 0; _i106 < 1; ++_i106) {
            t105[_i106] = 2.0f * t47[_i106];
        }
        float t107[1];
        for (uint _i108 = 0; _i108 < 1; ++_i108) {
            t107[_i108] = t23[_i108] + t105[_i108];
        }
        float t109[1];
        for (uint _i110 = 0; _i110 < 1; ++_i110) {
            t109[_i110] = 2.0f * t71[_i110];
        }
        float t111[1];
        for (uint _i112 = 0; _i112 < 1; ++_i112) {
            t111[_i112] = t107[_i112] + t109[_i112];
        }
        float t113[1];
        for (uint _i114 = 0; _i114 < 1; ++_i114) {
            t113[_i114] = t111[_i114] + t95[_i114];
        }
        float t115[1];
        for (uint _i116 = 0; _i116 < 1; ++_i116) {
            t115[_i116] = 0.0016666667070239782f * t113[_i116];
        }
        float t117[1];
        for (uint _i118 = 0; _i118 < 1; ++_i118) {
            t117[_i118] = t12[_i118] + t115[_i118];
        }
        float t119[1];
        for (uint _i120 = 0; _i120 < 1; ++_i120) {
            t119[_i120] = 2.0f * t55[_i120];
        }
        float t121[1];
        for (uint _i122 = 0; _i122 < 1; ++_i122) {
            t121[_i122] = t31[_i122] + t119[_i122];
        }
        float t123[1];
        for (uint _i124 = 0; _i124 < 1; ++_i124) {
            t123[_i124] = 2.0f * t79[_i124];
        }
        float t125[1];
        for (uint _i126 = 0; _i126 < 1; ++_i126) {
            t125[_i126] = t121[_i126] + t123[_i126];
        }
        float t127[1];
        for (uint _i128 = 0; _i128 < 1; ++_i128) {
            t127[_i128] = t125[_i128] + t103[_i128];
        }
        float t129[1];
        for (uint _i130 = 0; _i130 < 1; ++_i130) {
            t129[_i130] = 0.0016666667070239782f * t127[_i130];
        }
        float t131[1];
        for (uint _i132 = 0; _i132 < 1; ++_i132) {
            t131[_i132] = t14[_i132] + t129[_i132];
        }
        for (uint _i133 = 0; _i133 < 1; ++_i133) {
            t12[_i133] = t117[_i133];
        }
        for (uint _i134 = 0; _i134 < 1; ++_i134) {
            t14[_i134] = t131[_i134];
        }
    }
    for (uint _i135 = 0; _i135 < 1; ++_i135) {
        arg6[_i135] = t12[_i135];
    }
    for (uint _i136 = 0; _i136 < 1; ++_i136) {
        arg7[_i136] = t14[_i136];
    }
}
