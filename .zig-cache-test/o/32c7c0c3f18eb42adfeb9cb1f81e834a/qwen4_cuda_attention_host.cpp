#include "hc_native_support.h"
namespace ca {
constexpr int HD=256,JT=1;static uint3 blockIdx,threadIdx;
inline float tf_exp(float x){return expf(x); }
static bfloat probability(float score,float maximum,float& sum){
 const int jt=0,e=0,h=0,g=0,warp=0,t=0;const float s[1][4]={{score}},next_m[2]={maximum};
 float rows[2]={};bfloat ps[16][32];
                const float p = s[jt][e] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(s[jt][e], next_m[h]));
                rows[h] = __fadd_rn(rows[h], p);
                ps[g + 8 * h][warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] = __float2bfloat16_rn(p);
 sum=rows[0];return ps[0][0];}
extern "C" void fn_attn_merge(const float* __restrict__ PO, const float* __restrict__ PM,
                                                                const float* __restrict__ PL, const int* __restrict__ POS0,
                                                                __nv_bfloat16* __restrict__ OUT, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                int QSA) {
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    int n = POS0[0] + r + 1;
    if (QSA && SPR[r] != 0) n = NKR[r];
    const int chunks = (n + CH - 1) / CH;
    {
        float m = -INFINITY, l = 0.0f, o = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            const long long base = (static_cast<long long>(r) * NCH + c) * H + head;
            const float cm = PM[base], cl = PL[base], co = PO[base * HD + d];
            const bool active = cl > 0.0f;
            const float next = active ? fmaxf(m, cm) : m;
            const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
            o = __fmaf_rn(co, b, __fmul_rn(o, a));
            l = __fmaf_rn(cl, b, __fmul_rn(l, a));
            m = next;
        }
        OUT[(static_cast<long long>(r) * H + head) * HD + d] = __float2bfloat16_rn(__fdiv_rn(o, l));
    }
}static void run(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){
 if(count<1||count>32)std::abort();float scores[32],sum=0,maximum=-INFINITY; bfloat weights[32];
 for(int i=0;i<count;++i){float dot=0;for(int d=0;d<256;++d)dot=fmaf(float(reinterpret_cast<const bfloat*>(q)[d]),float(reinterpret_cast<const bfloat*>(k)[ids[i]*256+d]),dot);scores[i]=dot*0.0625f;maximum=fmaxf(maximum,scores[i]);}
 for(int i=0;i<count;++i){float term;weights[i]=probability(scores[i],maximum,term);sum+=term;}
 if(control)for(int i=0;i<count;++i)weights[i]=bfloat(expf(scores[i]-maximum)/sum);
 float po[256];for(int d=0;d<256;++d){po[d]=0;for(int i=0;i<count;++i)po[d]=fmaf(float(weights[i]),float(reinterpret_cast<const bfloat*>(v)[ids[i]*256+d]),po[d]);}
 const float pm=maximum,pl=control?1.0f:sum;const int pos=count-1;
 blockIdx={0,0,0};for(int d=0;d<256;++d){threadIdx={uint(d),0,0};fn_attn_merge(po,&pm,&pl,&pos,reinterpret_cast<bfloat*>(output),nullptr,nullptr,1,1,1,512,1,0);}
}
}
extern "C" void tf_q4_cuda_node_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output,int control){ca::run(q,k,v,ids,count,output,control);}
