import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
#from numba import jit,njit
import defns, F_mat, G_mat, K2i_mat
##############################################################
# Compute full matrix F3 for 2+1 systems
##############################################################
def F3mat_2plus1(E,L,nnP, f_qcot_1sp,f_qcot_2s, M12=[1,1],waves='sp',nnk_lists_12=None):
  M1,M2 = M12
  if nnk_lists_12==None:
    nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
    nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
    nnk_lists_12 = [nnk_list_1, nnk_list_2]

  F = F_mat.F_full_2plus1_scratch(E,nnP,L, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, diag_only=False)
  G = G_mat.Gmat_2plus1(E,L,nnP, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12)
  K2i = K2i_mat.K2_inv_mat_2plus1(E,L,nnP,f_qcot_1sp,f_qcot_2s, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, IPV=0)
  F3 = F/3 - F @ LA.inv(K2i + F + G) @ F
  return F3


##############################################################
# Compute full matrix F3 for ND systems (s-wave only)
##############################################################
def F3mat_ND(E,L,nnP, f_qcot_1s,f_qcot_2s,f_qcot_3s, M123=[1,1,1], nnk_lists_123=None):
  M1,M2,M3 = M123
  if nnk_lists_123==None:
    nnk_lists_123 = [defns.list_nnk_nnP(E,L,nnP, Mijk=defns.get_Mijk(M123,i)) for i in range(3)]

  F = F_mat.F_full_ND_scratch(E,nnP,L, M123=M123, nnk_lists_123=nnk_lists_123, diag_only=False)
  G = G_mat.Gmat_ND(E,L,nnP, M123=M123, nnk_lists_123=nnk_lists_123)
  K2i = K2i_mat.K2_inv_mat_ND(E,L,nnP,[f_qcot_1s,f_qcot_2s,f_qcot_3s], M123=M123, nnk_lists_123=nnk_lists_123, IPV=0)
  F3 = F/3 - F @ LA.inv(K2i + F + G) @ F
  return F3


##############################################################
# Compute full matrix F3 for ID systems (s-wave only)
##############################################################
def F3mat_ID(E,L,nnP, f_qcot_s, nnk_list=None):
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])
  F = F_mat.F_full_ID_scratch(E,nnP,L, nnk_list=nnk_list, diag_only=False)
  G = G_mat.Gmat_ID(E,L,nnP, nnk_list=nnk_list)
  K2i = K2i_mat.K2_inv_mat_ID(E,L,nnP,f_qcot_s, nnk_list=nnk_list, IPV=0)
  F3 = F/3 - F @ LA.inv(K2i + F + G) @ F
  return F3
