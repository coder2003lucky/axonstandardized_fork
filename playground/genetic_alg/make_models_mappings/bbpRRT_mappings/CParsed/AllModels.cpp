// Automatically generated C for /global/cscratch1/sd/zladd/axonstandardized/playground/genetic_alg/make_models_mappings/bbpRRT_mappings./runModel.hoc
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "AllModels.h"

// Universals:
#define PI (3.1415927f)
#define R (8.31441f)
#define FARADAY (96485.309f)
#define ktf (1000.*8.3134*(celsius + 273.15)/FARADAY)

// Reversals:
#define ena (50.0f)
#define DEF_cai		5.e-5	   /* mM */
#define DEF_cao		2.	   /* mM */
#define ek (-85.0f)

// Locals:
MYFTYPE N,P_a,P_b,a,b,gCa,gCa_LVAst,gIh,gIm,gK_Pst,gK_Tst,gNaTa_t,gNaTs2_t,gNap_Et2,gSK_E2,gSKv3_1,gk,hAlpha,hBeta,hInf,hTau,ihcn,mAlpha,mBeta,mInf,mTau,n0_n1_new,ninf,ntau,rng,scale_dens,zInf;

// Ion currents as Locals:
MYFTYPE ik,ica,ina;

// NGlobals:
#define vmin_StochKv (-120.0)
#define vmax_StochKv (100.0)
#define q10_StochKv (2.3)
#define temp_StochKv (23.0)
#define tadj_StochKv (2.4997732686681893)

// Declarations:
void rates_Ca(MYFTYPE v,MYFTYPE gCabar_Ca);
void rates_Ca_LVAst(MYFTYPE v,MYFTYPE gCa_LVAstbar_Ca_LVAst);
void rates_Ih(MYFTYPE v,MYFTYPE gIhbar_Ih,MYFTYPE ehcn_Ih);
void rates_Im(MYFTYPE v,MYFTYPE gImbar_Im);
void rates_K_Pst(MYFTYPE v,MYFTYPE gK_Pstbar_K_Pst);
void rates_K_Tst(MYFTYPE v,MYFTYPE gK_Tstbar_K_Tst);
void rates_Nap_Et2(MYFTYPE v,MYFTYPE gNap_Et2bar_Nap_Et2);
void rates_NaTa_t(MYFTYPE v,MYFTYPE gNaTa_tbar_NaTa_t);
void rates_NaTs2_t(MYFTYPE v,MYFTYPE gNaTs2_tbar_NaTs2_t);
void rates_SK_E2(MYFTYPE ca,MYFTYPE gSK_E2bar_SK_E2,MYFTYPE zTau_SK_E2);
void rates_SKv3_1(MYFTYPE v,MYFTYPE gSKv3_1bar_SKv3_1);
void states_StochKv(MYFTYPE v,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv);
void trates_StochKv(MYFTYPE v ,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv);
void ChkProb_StochKv(MYFTYPE p,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv);
void setRNG_StochKv(,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv);

// Functions:
MYFTYPE SigmoidRate_StochKv(MYFTYPE v ,MYFTYPE th ,MYFTYPE a (1/ms){
UNITSOFF;
if (fabs(v-th) > 1e-6 ) {;
return  a * (v - th) / (1 - exp(-(v - th)/q));
} else {;
return  a * q;
};
};
MYFTYPE strap_StochKv(MYFTYPE x){
if (x < 0) {;
return  0;
VERBATIM;
fprintf (stderr,"skv.mod;
ENDVERBATIM;
} else {;
return  x;
};
};
MYFTYPE urand_StochKv(MYFTYPE ){
VERBATIM;
/*;
;
;
;
*/;
double value;;
value = nrn_random_pick(_p_rng);;
return(value);;
ENDVERBATIM;
return  value;
};
MYFTYPE brand_StochKv(MYFTYPE P,MYFTYPE  N){
VERBATIM;
/*;
;
;
;
*/;
// Should probably be optimized;
double value = 0.0;;
int i;;
for (i = 0; i < _lN; i++) {;
if (nrn_random_pick(_p_rng) < _lP) {;
value = value + 1;;
};
};
return(value);;
ENDVERBATIM;
return  value;
};
MYFTYPE BnlDev _StochKv(MYFTYPE ppr,MYFTYPE  nnr){
VERBATIM;
int j;;
static int nold=(-1);;
double am,em,g,angle,p,bnl,sq,bt,y;;
static double pold=(-1.0),pc,plog,pclog,en,oldg;;
;
/* prepare to always ignore errors within this routine */;
;
;
p=(_lppr <= 0.5 ? _lppr;
am=_lnnr*p;;
if (_lnnr < 25) {;
bnl=0.0;;
for (j=1;j<=_lnnr;j++);
if (urand_StochKv() < p) bnl += 1.0;;
};
else if (am < 1.0) {;
g=exp(-am);;
bt=1.0;;
for (j=0;j<=_lnnr;j++) {;
bt *= urand_StochKv();;
if (bt < g) break;;
};
bnl=(j <= _lnnr ? j;
};
else {;
if (_lnnr != nold) {;
en=_lnnr;;
oldg=gammln(en+1.0);;
nold=_lnnr;;
};
if (p != pold) {;
pc=1.0-p;;
plog=log(p);;
pclog=log(pc);;
pold=p;;
};
sq=sqrt(2.0*am*pc);;
do {;
do {;
angle=PI*urand_StochKv();;
angle=PI*urand_StochKv();;
y=tan(angle);;
em=sq*y+am;;
} while (em < 0.0 || em >= (en+1.0));;
em=floor(em);;
bt=1.2*sq*(1.0+y*y)*exp(oldg-gammln(em+1.0) -;
gammln(en-em+1.0)+em*plog+(en-em)*pclog);;
} while (urand_StochKv() > bt);;
bnl=em;;
};
if (p != _lppr) bnl=_lnnr-bnl;;
;
/* recover error if changed during this routine, thus ignoring;
any errors during this routine */;
;
;
return bnl;;
;
ENDVERBATIM;
return  bnl;
};

// Inits:


void InitModel_Ca(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gCabar_Ca, MYFTYPE &ica,MYFTYPE &eca, MYFTYPE &cai){
eca = ktf/2 *log(DEF_cao / cai);
	rates_Ca(,gCabar_Ca);
	m = mInf;
	h = hInf;
};


void InitModel_Ca_LVAst(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gCa_LVAstbar_Ca_LVAst, MYFTYPE &ica,MYFTYPE &eca, MYFTYPE &cai){
eca = ktf/2 *log(DEF_cao / cai);
	rates_Ca_LVAst(,gCa_LVAstbar_Ca_LVAst);
	m = mInf;
	h = hInf;
};


void InitModel_CaDynamics_E2(MYFTYPE v,MYFTYPE &cai,MYFTYPE gamma_CaDynamics_E2,MYFTYPE decay_CaDynamics_E2,MYFTYPE depth_CaDynamics_E2,MYFTYPE minCai_CaDynamics_E2, MYFTYPE ica,MYFTYPE &eca){
cai = DEF_cai;
eca = ktf/2 *log(DEF_cao / cai);
};


void InitModel_Ih(MYFTYPE v,MYFTYPE &m,MYFTYPE gIhbar_Ih,MYFTYPE ehcn_Ih){
	rates_Ih(,gIhbar_Ih,ehcn_Ih);
	m = mInf;
};


void InitModel_Im(MYFTYPE v,MYFTYPE &m,MYFTYPE gImbar_Im){
	rates_Im(,gImbar_Im);
	m = mInf;
};


void InitModel_K_Pst(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gK_Pstbar_K_Pst){
	rates_K_Pst(,gK_Pstbar_K_Pst);
	m = mInf;
	h = hInf;
};


void InitModel_K_Tst(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gK_Tstbar_K_Tst){
	rates_K_Tst(,gK_Tstbar_K_Tst);
	m = mInf;
	h = hInf;
};


void InitModel_Nap_Et2(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNap_Et2bar_Nap_Et2){
	rates_Nap_Et2(,gNap_Et2bar_Nap_Et2);
	m = mInf;
	h = hInf;
};


void InitModel_NaTa_t(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNaTa_tbar_NaTa_t){
	rates_NaTa_t(,gNaTa_tbar_NaTa_t);
	m = mInf;
	h = hInf;
};


void InitModel_NaTs2_t(MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNaTs2_tbar_NaTs2_t){
	rates_NaTs2_t(,gNaTs2_tbar_NaTs2_t);
	m = mInf;
	h = hInf;
};


void InitModel_pas(MYFTYPE v,MYFTYPE g_pas,MYFTYPE e_pas){
};


void InitModel_SK_E2(MYFTYPE v,MYFTYPE &z,MYFTYPE gSK_E2bar_SK_E2,MYFTYPE zTau_SK_E2, MYFTYPE cai,MYFTYPE &eca){
eca = ktf/2 *log(DEF_cao / cai);
        rates_SK_E2(cai,gSK_E2bar_SK_E2,zTau_SK_E2);
        z = zInf;
};


void InitModel_SKv3_1(MYFTYPE v,MYFTYPE &m,MYFTYPE gSKv3_1bar_SKv3_1){
	rates_SKv3_1(,gSKv3_1bar_SKv3_1);
	m = mInf;
};


void InitModel_StochKv(MYFTYPE v,MYFTYPE &n,MYFTYPE &N0,MYFTYPE &n0_n1,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv){
    eta = gkbar_StochKv / gamma_StochKv;
    trates_StochKv(v,gamma_StochKv,gkbar_StochKv,tha_StochKv,qa_StochKv,Ra_StochKv,Rb_StochKv,deterministic_StochKv);
    n = ninf;
    scale_dens = gamma_StochKv/area;
    N = floor(eta*area + 0.5);
    ;
    N1 = floor(n * N + 0.5);
    N0 = N-N1       ;
    ;
    n0_n1 = 0;
    n1_n0 = 0;
};

// Procedures:
void rates_Ca(MYFTYPE v,MYFTYPE gCabar_Ca) {
    if ( ( v  == - 27.0 ) ) {
     v = v + 0.0001 ;
     }
   mAlpha = ( 0.055 * ( - 27.0 - v ) ) / ( exp ( ( - 27.0 - v ) / 3.8 ) - 1.0 ) ;
   mBeta = ( 0.94 * exp ( ( - 75.0 - v ) / 17.0 ) ) ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   mTau = 1.0 / ( mAlpha + mBeta ) ;
   hAlpha = ( 0.000457 * exp ( ( - 13.0 - v ) / 50.0 ) ) ;
   hBeta = ( 0.0065 / ( exp ( ( - v - 15.0 ) / 28.0 ) + 1.0 ) ) ;
   hInf = hAlpha / ( hAlpha + hBeta ) ;
   hTau = 1.0 / ( hAlpha + hBeta ) ;
}
void rates_Ca_LVAst(MYFTYPE v,MYFTYPE gCa_LVAstbar_Ca_LVAst) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    v = v + 10.0 ;
   mInf = 1.0000 / ( 1.0 + exp ( ( v - - 30.000 ) / - 6.0 ) ) ;
   mTau = ( 5.0000 + 20.0000 / ( 1.0 + exp ( ( v - - 25.000 ) / 5.0 ) ) ) / qt ;
   hInf = 1.0000 / ( 1.0 + exp ( ( v - - 80.000 ) / 6.4 ) ) ;
   hTau = ( 20.0000 + 50.0000 / ( 1.0 + exp ( ( v - - 40.000 ) / 7.0 ) ) ) / qt ;
   v = v - 10.0 ;
}
void rates_Ih(MYFTYPE v,MYFTYPE gIhbar_Ih,MYFTYPE ehcn_Ih) {
    if ( v  == - 154.9 ) {
     v = v + 0.0001 ;
     }
   mAlpha = 0.001 * 6.43 * ( v + 154.9 ) / ( exp ( ( v + 154.9 ) / 11.9 ) - 1.0 ) ;
   mBeta = 0.001 * 193.0 * exp ( v / 33.1 ) ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   mTau = 1.0 / ( mAlpha + mBeta ) ;
}
void rates_Im(MYFTYPE v,MYFTYPE gImbar_Im) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    mAlpha = 3.3e-3 * exp ( 2.5 * 0.04 * ( v - - 35.0 ) ) ;
   mBeta = 3.3e-3 * exp ( - 2.5 * 0.04 * ( v - - 35.0 ) ) ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   mTau = ( 1.0 / ( mAlpha + mBeta ) ) / qt ;
}
void rates_K_Pst(MYFTYPE v,MYFTYPE gK_Pstbar_K_Pst) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    v = v + 10.0 ;
   mInf = ( 1.0 / ( 1.0 + exp ( - ( v + 1.0 ) / 12.0 ) ) ) ;
   if ( v < - 50.0 ) {
     mTau = ( 1.25 + 175.03 * exp ( - v * - 0.026 ) ) / qt ;
     }
   else {
     mTau = ( ( 1.25 + 13.0 * exp ( - v * 0.026 ) ) ) / qt ;
     }
   hInf = 1.0 / ( 1.0 + exp ( - ( v + 54.0 ) / - 11.0 ) ) ;
   hTau = ( 360.0 + ( 1010.0 + 24.0 * ( v + 55.0 ) ) * exp ( - powf( ( ( v + 75.0 ) / 48.0 ) , 2.0 ) ) ) / qt ;
   v = v - 10.0 ;
}
void rates_K_Tst(MYFTYPE v,MYFTYPE gK_Tstbar_K_Tst) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    v = v + 10.0 ;
   mInf = 1.0 / ( 1.0 + exp ( - ( v + 0.0 ) / 19.0 ) ) ;
   mTau = ( 0.34 + 0.92 * exp ( - powf( ( ( v + 71.0 ) / 59.0 ) , 2.0 ) ) ) / qt ;
   hInf = 1.0 / ( 1.0 + exp ( - ( v + 66.0 ) / - 10.0 ) ) ;
   hTau = ( 8.0 + 49.0 * exp ( - powf( ( ( v + 73.0 ) / 23.0 ) , 2.0 ) ) ) / qt ;
   v = v - 10.0 ;
}
void rates_Nap_Et2(MYFTYPE v,MYFTYPE gNap_Et2bar_Nap_Et2) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    mInf = 1.0 / ( 1.0 + exp ( ( v - - 52.6 ) / - 4.6 ) ) ;
   if ( v  == - 38.0 ) {
     v = v + 0.0001 ;
     }
   mAlpha = ( 0.182 * ( v - - 38.0 ) ) / ( 1.0 - ( exp ( - ( v - - 38.0 ) / 6.0 ) ) ) ;
   mBeta = ( 0.124 * ( - v - 38.0 ) ) / ( 1.0 - ( exp ( - ( - v - 38.0 ) / 6.0 ) ) ) ;
   mTau = 6.0 * ( 1.0 / ( mAlpha + mBeta ) ) / qt ;
   if ( v  == - 17.0 ) {
     v = v + 0.0001 ;
     }
   if ( v  == - 64.4 ) {
     v = v + 0.0001 ;
     }
   hInf = 1.0 / ( 1.0 + exp ( ( v - - 48.8 ) / 10.0 ) ) ;
   hAlpha = - 2.88e-6 * ( v + 17.0 ) / ( 1.0 - exp ( ( v + 17.0 ) / 4.63 ) ) ;
   hBeta = 6.94e-6 * ( v + 64.4 ) / ( 1.0 - exp ( - ( v + 64.4 ) / 2.63 ) ) ;
   hTau = ( 1.0 / ( hAlpha + hBeta ) ) / qt ;
}
void rates_NaTa_t(MYFTYPE v,MYFTYPE gNaTa_tbar_NaTa_t) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    if ( v  == - 38.0 ) {
     v = v + 0.0001 ;
     }
   mAlpha = ( 0.182 * ( v - - 38.0 ) ) / ( 1.0 - ( exp ( - ( v - - 38.0 ) / 6.0 ) ) ) ;
   mBeta = ( 0.124 * ( - v - 38.0 ) ) / ( 1.0 - ( exp ( - ( - v - 38.0 ) / 6.0 ) ) ) ;
   mTau = ( 1.0 / ( mAlpha + mBeta ) ) / qt ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   if ( v  == - 66.0 ) {
     v = v + 0.0001 ;
     }
   hAlpha = ( - 0.015 * ( v - - 66.0 ) ) / ( 1.0 - ( exp ( ( v - - 66.0 ) / 6.0 ) ) ) ;
   hBeta = ( - 0.015 * ( - v - 66.0 ) ) / ( 1.0 - ( exp ( ( - v - 66.0 ) / 6.0 ) ) ) ;
   hTau = ( 1.0 / ( hAlpha + hBeta ) ) / qt ;
   hInf = hAlpha / ( hAlpha + hBeta ) ;
}
void rates_NaTs2_t(MYFTYPE v,MYFTYPE gNaTs2_tbar_NaTs2_t) {
   MYFTYPE qt ;
 qt = powf( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    if ( v  == - 32.0 ) {
     v = v + 0.0001 ;
     }
   mAlpha = ( 0.182 * ( v - - 32.0 ) ) / ( 1.0 - ( exp ( - ( v - - 32.0 ) / 6.0 ) ) ) ;
   mBeta = ( 0.124 * ( - v - 32.0 ) ) / ( 1.0 - ( exp ( - ( - v - 32.0 ) / 6.0 ) ) ) ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   mTau = ( 1.0 / ( mAlpha + mBeta ) ) / qt ;
   if ( v  == - 60.0 ) {
     v = v + 0.0001 ;
     }
   hAlpha = ( - 0.015 * ( v - - 60.0 ) ) / ( 1.0 - ( exp ( ( v - - 60.0 ) / 6.0 ) ) ) ;
   hBeta = ( - 0.015 * ( - v - 60.0 ) ) / ( 1.0 - ( exp ( ( - v - 60.0 ) / 6.0 ) ) ) ;
   hInf = hAlpha / ( hAlpha + hBeta ) ;
   hTau = ( 1.0 / ( hAlpha + hBeta ) ) / qt ;
}
void rates_SK_E2(MYFTYPE ca,MYFTYPE gSK_E2bar_SK_E2,MYFTYPE zTau_SK_E2) {
   if ( ca < 1e-7 ) {
     ca = ca + 1e-07 ;
     }
   zInf = 1.0 / ( 1.0 + powf( ( 0.00043 / ca ) , 4.8 ) ) ;
}
void rates_SKv3_1(MYFTYPE v,MYFTYPE gSKv3_1bar_SKv3_1) {
    mInf = 1.0 / ( 1.0 + exp ( ( ( v - ( 18.700 ) ) / ( - 9.700 ) ) ) ) ;
   mTau = 0.2 * 20.000 / ( 1.0 + exp ( ( ( v - ( - 46.560 ) ) / ( - 44.140 ) ) ) ) ;
}
void states_StochKv(MYFTYPE v,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv) {
   trates_StochKv (   v ,gamma_StochKv,gkbar_StochKv,tha_StochKv,qa_StochKv,Ra_StochKv,Rb_StochKv,deterministic_StochKv) ;
   P_a = strap_StochKv (   a * dt ) ;
   P_b = strap_StochKv (   b * dt ) ;
   ChkProb_StochKv (   P_a ,gamma_StochKv,gkbar_StochKv,tha_StochKv,qa_StochKv,Ra_StochKv,Rb_StochKv,deterministic_StochKv) ;
   ChkProb_StochKv (   P_b ,gamma_StochKv,gkbar_StochKv,tha_StochKv,qa_StochKv,Ra_StochKv,Rb_StochKv,deterministic_StochKv) ;
   n0_n1 = BnlDev _StochKv(   P_a , N0 ) ;
   n1_n0 = BnlDev _StochKv(   P_b , N1 ) ;
   N0 = strap_StochKv (   N0 - n0_n1 + n1_n0 ) ;
   N1 = N - N0 ;
}
void trates_StochKv(MYFTYPE v ,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv) {
   tadj_StochKv = powf( q10_StochKv , ( ( celsius - temp_StochKv ) / ( 10.0 ) ) ) ;
   a = SigmoidRate_StochKv (   v , tha_StochKv , Ra_StochKv , qa_StochKv ) ;
   a = a * tadj_StochKv ;
   b = SigmoidRate_StochKv (   - v , - tha_StochKv , Rb_StochKv , qa_StochKv ) ;
   b = b * tadj_StochKv ;
   ntau = 1.0 / ( a + b ) ;
   ninf = a * ntau ;
}
void ChkProb_StochKv(MYFTYPE p,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv) {
   if ( p < 0.0  || p > 1.0 ) {
     
/*VERBATIM*/
// ToDo: should be disabled during ForwardSkip and enabled right after
//    fprintf(stderr, "StochKv.mod:ChkProb_StochKv: argument not a probability.\n");
 }
}
void setRNG_StochKv(,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv) {
   
/*VERBATIM*/
    {
        /**
         * This function takes a NEURON Random object declared in hoc and makes it usable by this mod file.
         * Note that this method is taken from Brett paper as used by netstim.hoc and netstim.mod
         * which points out that the Random must be in negexp(1) mode
         */
        void** pv = (void**)(&_p_rng);
        if( ifarg(1)) {
            *pv = nrn_random_arg(1);
        } else {
            *pv = (void*)0;
        }
    }
}

// Derivs:

// Breaks:


void BreakpointModel_Ca(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gCabar_Ca, MYFTYPE &ica,MYFTYPE &eca, MYFTYPE &cai) {
MYFTYPE ica_Ca;

   gCa = gCabar_Ca * m * m * h ;
   ica_Ca = gCa * ( v - eca ) ;
sumCurrents+= ica_Ca;

 ica += ica_Ca;
sumConductivity+= gCa;
};


void BreakpointModel_Ca_LVAst(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gCa_LVAstbar_Ca_LVAst, MYFTYPE &ica,MYFTYPE &eca, MYFTYPE &cai) {
MYFTYPE ica_Ca_LVAst;

   gCa_LVAst = gCa_LVAstbar_Ca_LVAst * m * m * h ;
   ica_Ca_LVAst = gCa_LVAst * ( v - eca ) ;
sumCurrents+= ica_Ca_LVAst;

 ica += ica_Ca_LVAst;
sumConductivity+= gCa_LVAst;
};


void BreakpointModel_CaDynamics_E2(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &cai,MYFTYPE gamma_CaDynamics_E2,MYFTYPE decay_CaDynamics_E2,MYFTYPE depth_CaDynamics_E2,MYFTYPE minCai_CaDynamics_E2, MYFTYPE ica,MYFTYPE &eca) {
};


void BreakpointModel_Ih(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE gIhbar_Ih,MYFTYPE ehcn_Ih) {
   gIh = gIhbar_Ih * m ;
   ihcn = gIh * ( v - ehcn_Ih ) ;
i = ihcn;

sumCurrents+= i;
sumConductivity+= gIh;
};


void BreakpointModel_Im(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE gImbar_Im) {
   gIm = gImbar_Im * m ;
   ik = gIm * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gIm;
};


void BreakpointModel_K_Pst(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gK_Pstbar_K_Pst) {
   gK_Pst = gK_Pstbar_K_Pst * m * m * h ;
   ik = gK_Pst * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gK_Pst;
};


void BreakpointModel_K_Tst(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gK_Tstbar_K_Tst) {
   gK_Tst = gK_Tstbar_K_Tst * powf( m , 4.0 ) * h ;
   ik = gK_Tst * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gK_Tst;
};


void BreakpointModel_Nap_Et2(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNap_Et2bar_Nap_Et2) {
   gNap_Et2 = gNap_Et2bar_Nap_Et2 * m * m * m * h ;
   ina = gNap_Et2 * ( v - ena ) ;
sumCurrents+= ina;
sumConductivity+= gNap_Et2;
};


void BreakpointModel_NaTa_t(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNaTa_tbar_NaTa_t) {
   gNaTa_t = gNaTa_tbar_NaTa_t * m * m * m * h ;
   ina = gNaTa_t * ( v - ena ) ;
sumCurrents+= ina;
sumConductivity+= gNaTa_t;
};


void BreakpointModel_NaTs2_t(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE &h,MYFTYPE gNaTs2_tbar_NaTs2_t) {
   gNaTs2_t = gNaTs2_tbar_NaTs2_t * m * m * m * h ;
   ina = gNaTs2_t * ( v - ena ) ;
sumCurrents+= ina;
sumConductivity+= gNaTs2_t;
};


void BreakpointModel_pas(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE g_pas,MYFTYPE e_pas) {
   i = g_pas * ( v - e_pas ) ;
i = i;

sumCurrents+= i;
sumConductivity+= g_pas;
};


void BreakpointModel_SK_E2(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &z,MYFTYPE gSK_E2bar_SK_E2,MYFTYPE zTau_SK_E2, MYFTYPE cai,MYFTYPE &eca) {
   gSK_E2 = gSK_E2bar_SK_E2 * z ;
   ik = gSK_E2 * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gSK_E2;
};


void BreakpointModel_SKv3_1(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &m,MYFTYPE gSKv3_1bar_SKv3_1) {
   gSKv3_1 = gSKv3_1bar_SKv3_1 * m ;
   ik = gSKv3_1 * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gSKv3_1;
};


void BreakpointModel_StochKv(MYSECONDFTYPE &sumCurrents, MYFTYPE &sumConductivity, MYFTYPE v,MYFTYPE &n,MYFTYPE &N0,MYFTYPE &n0_n1,MYFTYPE gamma_StochKv,MYFTYPE gkbar_StochKv,MYFTYPE tha_StochKv,MYFTYPE qa_StochKv,MYFTYPE Ra_StochKv,MYFTYPE Rb_StochKv,MYFTYPE deterministic_StochKv) {
   gk = ( strap_StochKv (   N1 ,gamma_StochKv,gkbar_StochKv,tha_StochKv,qa_StochKv,Ra_StochKv,Rb_StochKv,deterministic_StochKv) * scale_dens * tadj_StochKv ) ;
   ik = 1e-4 * gk * ( v - ek ) ;
sumCurrents+= ik;
sumConductivity+= gk;
};
