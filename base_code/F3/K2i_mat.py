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
    out = eta_i/(16*pi*omk*E2kstar) * ( qcot0 + q_abs*(1-h)) - IPV*h/(64*pi*omk)
  elif l==1 and -1<=m<=1:
    q3cot1 = f_qcot_l(q2)
    out = eta_i/(16*pi*omk*E2kstar) * ( q3cot1 + q2*q_abs*(1-h) ) - q2*IPV*h/(64*pi*omk) # Q matrices already applied
  elif l==2 and -2<=m<=2:
    q5cot2 = f_qcot_l(q2)
    out = eta_i/(16*pi*omk*E2kstar) * ( q5cot2 + q2**2*q_abs*(1-h) ) - q2**2*h*IPV/(64*pi*omk) # Q matrices applied
  else:
    return 0

  if out.imag > 1e-15:
    sys.error('Error in K2i_inv: imaginary part in output')
    raise ValueError
  else:
    out = out.real
  return out

################################################################################
# Compute WxW matrix K2i_inv/(2*omega_k), no L^3, for a given kvec (W=4 for waves='sp')
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
  elif waves=='sd':
    f_qcot_i_d = f_qcot_i_waves[1]
    K2i_dwave = K2i_inv(E,kvec,2,0,Pvec, f_qcot_i_d, Mijk=Mijk,eta_i=eta_i,IPV=IPV)
    #print(Pvec,kvec,K2i_dwave)
    K2i_k_diag = [K2i_swave] + 5*[K2i_dwave]
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
  return block_diag(*K2_list) / L**3  # 1/L^3 factor to match convention of TOPT papers

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
def K2_inv_mat_ND(E,L,nnP,f_qcot_123, M123=[1,1,1], nnk_lists_123=None, IPV=0):
  waves = 's'  # only use s-wave for ND case (Kdf3 isn't ready for p-wave)
  [M1,M2,M3] = M123
  if nnk_lists_123==None:
    nnk_lists_123 = []
    for i in range(3):
      nnk_lists_123.append(defns.list_nnk_nnP(E,L,nnP, Mijk=defns.get_Mijk(M123,i)))
  K2L_inv_list = []
  for i in range(3):
    Mijk = defns.get_Mijk(M123,i)
    K2L_inv_list.append(K2i_inv_mat(E,L,nnP, f_qcot_123[i], Mijk=Mijk, waves=waves, eta_i=1, nnk_list=nnk_lists_123[i], IPV=IPV) )
  return block_diag(*K2L_inv_list)

################################################################################
# Compute full ID matrix \overline{K}_{2,L}^{-1}
################################################################################
def K2_inv_mat_ID(E,L,nnP,f_qcot_waves, nnk_list=None, IPV=0):
  waves = 's'   # only use s-wave for ID case (Kdf3 isn't ready for d-wave)
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])
  K2L_inv = K2i_inv_mat(E,L,nnP, f_qcot_waves, Mijk=[1,1,1], waves=waves, eta_i=0.5, nnk_list=nnk_list, IPV=IPV)
  return K2L_inv

################################################################################
# Compute 2-pt. ND & ID matrices K_2^{-1}
################################################################################
# ND case
def K2_inv_mat_2pt_ND(E2,P2vec,f_qcot_waves, waves='sp', IPV=0):
  K2i = 2*L**3 * K2i_mat.K2i_inv_k(E2+1,[0,0,0],P2vec,f_qcot_waves, Mijk=[1,M1,M2], waves=waves, eta_i=1, IPV=0)

# ID case (s-wave only)
def K2_inv_mat_2pt_ID(E2,P2vec,f_qcot_s, IPV=0):
  K2i = 2*L**3 * K2i_mat.K2i_inv_k(E2+1,[0,0,0],P2vec,f_qcot_s, Mijk=[1,1,1], waves='s', eta_i=0.5, IPV=0)
