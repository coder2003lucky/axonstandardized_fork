#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _branching_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," neuron_files/allen//CaDynamics_E2.mod");
    fprintf(stderr," neuron_files/allen//Ca_HVA.mod");
    fprintf(stderr," neuron_files/allen//Ca_LVAst.mod");
    fprintf(stderr," neuron_files/allen//Ih.mod");
    fprintf(stderr," neuron_files/allen//Im.mod");
    fprintf(stderr," neuron_files/allen//K_Pst.mod");
    fprintf(stderr," neuron_files/allen//K_Tst.mod");
    fprintf(stderr," neuron_files/allen//NaTa_t.mod");
    fprintf(stderr," neuron_files/allen//NaTs2_t.mod");
    fprintf(stderr," neuron_files/allen//Nap_Et2.mod");
    fprintf(stderr," neuron_files/allen//SK_E2.mod");
    fprintf(stderr," neuron_files/allen//SKv3_1.mod");
    fprintf(stderr," neuron_files/allen//branching.mod");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _branching_reg();
}
