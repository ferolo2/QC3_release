import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
from scipy.linalg import block_diag
import defns
# from numba import jit,njit

################################################################################
# Calculate matrix element of K2i_inv/(2*omega), no L^3
################################################################################
#@njit(fastmath=True,cache=True)
def K2i_inv(E,kvec,l,m,Pvec,f_qcot_l, Mijk=[1,1,1], eta_i=1, IPV=0):
  [Mi,Mj,Mk] = Mijk
  k = LA.norm(kvec)
  omk = defns.omega(k,m=Mi)
  sig_i = defns.sigma_i(E,Pvec,kvec, Mi=Mi)
  E2kstar = sqrt(sig_i)
  q2 = defns.qst2_i(E,Pvec,kvec, Mijk=Mijk)
  q_abs = sqrt(abs(q2))
  h = defns.hh(sig_i,Mjk=[Mj,Mk])
  if h==0:
    return 0.
  if l==m==0:
    qcot0 = f_qcot_l(q2)
    out = eta_i/(8*pi*E2kstar*2*omk) * ( qcot0 + q_abs*(1-h)) - IPV*h/(32*pi*2*omk)
  elif l==1 and -1<=m<=1:
    q3cot1 = f_qcot_l(q2)
    out = eta_i/(8*pi*E2kstar*2*omk) * ( q3cot1 + q2*q_abs*(1-h) ) - q2*IPV*h/(32*pi*2*omk) # TB, no q
  else:
    return 0

  if out.imag > 1e-15:
    sys.error('Error in K2i_inv: imaginary part in output')
    raise ValueError
  else:
    out = out.real
  return out

################################################################################
# Compute 4x4 matrix K2i_inv/(2*omega_k), no L^3, for a given kvec (assumes waves='sp')
################################################################################
def K2i_inv_k(E,kvec,Pvec,f_qcot_i_waves, Mijk=[1,1,1], waves='sp', eta_i=1, IPV=0):
  f_qcot_i_s = f_qcot_i_waves[0]
  K2i_swave = K2i_inv(E,kvec,0,0,Pvec, f_qcot_i_s, Mijk=Mijk,eta_i=eta_i,IPV=IPV)

  if waves=='s':
    return np.array([K2i_swave])
  elif waves=='sp':
    f_qcot_i_p = f_qcot_i_waves[1]
    K2i_pwave = K2i_inv(E,kvec,1,0,Pvec, f_qcot_i_p, Mijk=Mijk,eta_i=eta_i,IPV=IPV)
    K2i_k_diag = [K2i_swave] + 3*[K2i_pwave]
    return np.diag(K2i_k_diag)

################################################################################
# Compute block matrix K2i_inv/(2*omega_k*L**3)
################################################################################
def K2i_inv_mat(E,L,nnP, f_qcot_i_waves, Mijk=[1,1,1], waves='sp', eta_i=1, nnk_list=None, IPV=0):
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
  N = len(nnk_list)
  Pvec = [2*pi/L*x for x in nnP]
  K2_list = []
  for k in range(N):
    kvec = [2*pi/L*x for x in nnk_list[k]]
    K2_list.append(K2i_inv_k(E,kvec,Pvec, f_qcot_i_waves, Mijk=Mijk, waves=waves, eta_i=eta_i,IPV=IPV))
  return block_diag(*K2_list) / L**3

################################################################################
# Compute full 2+1 matrix \overline{K}_{2,L}^{-1}
################################################################################
def K2_inv_mat_2plus1(E,L,nnP, f_qcot_1sp, f_qcot_2s, M12=[1,1], waves='sp', nnk_lists_12=None, IPV=0):
  [M1,M2] = M12
  if nnk_lists_12==None:
    nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
    nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
  else:
    nnk_list_1, nnk_list_2 = nnk_lists_12
  K2L_inv_1 = K2i_inv_mat(E,L,nnP, f_qcot_1sp, Mijk=[M1,M1,M2], waves=waves, eta_i=1, nnk_list=nnk_list_1, IPV=IPV)
  K2L_inv_2 = K2i_inv_mat(E,L,nnP, f_qcot_2s, Mijk=[M2,M1,M1], waves='s', eta_i=0.5, nnk_list=nnk_list_2, IPV=IPV)
  return block_diag(*[K2L_inv_1, 2*K2L_inv_2])

################################################################################
# Compute full ND matrix \overline{K}_{2,L}^{-1}
################################################################################
# def K2_inv_mat_ND(E,L,nnP,f_kcot0,f_kcot1, M123=[1,1,1], nnk_lists_123=None, IPV=0):
#   [M1,M2,M3] = M123
#   if nnk_lists_123==None:
#     nnk_lists_123 = []
#     for i in range(3):
#       nnk_lists_123.append(defns.list_nnk_nnP(E,L,nnP, Mijk=defns.get_Mijk(M123,i)))
#   K2L_inv_list = []
#   for i in range(3):
#     Mijk = defns.get_Mijk(M123,i)
#     K2L_inv_list.append(K2i_inv_mat(E,L,nnP,f_kcot0,f_kcot1, Mijk=Mijk, eta_i=1, nnk_list=nnk_lists_123[i], IPV=IPV))
#   return block_diag(*K2L_inv_list)
