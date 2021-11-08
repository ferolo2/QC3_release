import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
#from numba import jit,njit
import defns, F_fast, G_mov, K2i_mat
##############################################################
# Compute full matrix F3 for 2+1 systems
##############################################################

def F3mat_2plus1(E,L,nnP, f_qcot_1sp,f_qcot_2s, M12=[1,1],waves='sp',nnk_lists_12=None):
  M1,M2 = M12
  if nnk_lists_12==None:
    nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
    nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
    nnk_lists_12 = [nnk_list_1, nnk_list_2]

  F = F_fast.F_full_2plus1_scratch(E,nnP,L, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, diag_only=False)
  G = G_mov.Gmat_2plus1(E,L,nnP, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12)
  K2i = K2i_mat.K2_inv_mat_2plus1(E,L,nnP,f_qcot_1sp,f_qcot_2s, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, IPV=0)
  F3 = F/3 - F @ LA.inv(K2i + F + G) @ F
  return F3
