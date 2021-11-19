import numpy as np
#sqrt=np.sqrt
pi=np.pi; conj=np.conjugate; LA=np.linalg;
import defns
sqrt=defns.sqrt
#################################################################
# Want to compute K3E term in Kdf3 for 2+1 systems
#################################################################
# Compute 4x4 block of K3E for given (i,pvec;j,kvec)
def K3E_ij_pk(E,Pvec,pvec,kvec, i,j, M12=[1,1], waves_ij=('sp','sp')):
  M1,M2 = M12; Mtot = 2*M1+M2
  Mi,Mj = [M12[i-1],M12[j-1]]
  if i==j==1:
    Mk = M2
  else:
    Mk = M1

  Ecm2 = E**2 - sum([x**2 for x in Pvec])
  P2p_vec = Pvec-pvec; P2k_vec = Pvec-kvec

  om_pi = sqrt(sum([x**2 for x in pvec])+Mi**2)
  om_kj = sqrt(sum([x**2 for x in kvec])+Mj**2)

  E2p = E-om_pi; E2k = E-om_kj

  sig_pi = defns.sigma_i(E,Pvec,pvec,Mi=Mi)
  sig_kj = defns.sigma_i(E,Pvec,kvec,Mi=Mj)

  q2_pi = defns.lambda_tri(sig_pi,Mj**2,Mk**2)/(4*sig_pi)
  q2_kj = defns.lambda_tri(sig_kj,Mi**2,Mk**2)/(4*sig_kj)


  om_qpi_1 = sqrt(q2_pi + M1**2); om_qpi_2 = sqrt(q2_pi + M2**2)
  om_qkj_1 = sqrt(q2_kj + M1**2); om_qkj_2 = sqrt(q2_kj + M2**2)

  pms_0 = om_qpi_1 - om_qpi_2
  kms_0 = om_qkj_1 - om_qkj_2

  #om_p1

  psk_vec = defns.boost(om_pi,pvec,E2k,P2k_vec)
  ksp_vec = defns.boost(om_kj,kvec,E2p,P2p_vec)

  om_psk = sqrt(sum([x**2 for x in psk_vec]) + Mi**2)
  om_ksp = sqrt(sum([x**2 for x in ksp_vec]) + Mj**2)

  P2psk_vec = defns.boost(E2p,P2p_vec,E2k,P2k_vec)
  P2ksp_vec = defns.boost(E2k,P2k_vec,E2p,P2p_vec)

  P2psk_0 = sqrt( sig_pi + sum([x**2 for x in P2psk_vec]) )
  P2ksp_0 = sqrt( sig_kj + sum([x**2 for x in P2ksp_vec]) )

  beta_p_vec = np.array([x/E2p for x in P2p_vec]); gam_p = sqrt(1/(1-sum([x**2 for x in beta_p_vec])))
  beta_k_vec = np.array([x/E2k for x in P2k_vec]); gam_k = sqrt(1/(1-sum([x**2 for x in beta_k_vec])))
  beta_p = LA.norm(beta_p_vec)
  beta_k = LA.norm(beta_k_vec)
  beta_p_hat = np.array([x/beta_p for x in beta_p_vec]) if beta_p!=0 else beta_p_vec
  beta_k_hat = np.array([x/beta_k for x in beta_k_vec]) if beta_k!=0 else beta_k_vec

  V = -pms_0*gam_p * (beta_p_vec + beta_k_hat*np.dot(beta_k_hat,beta_p_vec)*(gam_k-1) - beta_k_vec*gam_k)
  Vp = -kms_0*gam_k * (beta_k_vec + beta_p_hat*np.dot(beta_p_hat,beta_k_vec)*(gam_p-1) - beta_p_vec*gam_p)

  t = np.zeros((3,3))
  for I in range(3):
    Im = (I-1) % 3  # convert Cartesian basis to spherical: I=(0,1,2)<->(x,y,z) --> m=(+1,-1,0)<->(2,1,0)=Im
    t[Im,Im] = -1
    for J in range(3):
      Jm = (J-1) % 3
      t[Im,Jm] += beta_p_hat[I]*beta_k_hat[J] * ( gam_k*beta_k*gam_p*beta_p - np.dot(beta_k_hat,beta_p_hat)*(gam_k-1)*(gam_p-1) ) - beta_k_hat[I]*beta_k_hat[J]*(gam_k-1) - beta_p_hat[I]*beta_p_hat[J]*(gam_p-1)


  Wi = defns.get_lm_size(waves_ij[0])
  Wj = defns.get_lm_size(waves_ij[1])
  out = np.zeros((Wi,Wj))

  if i==j==2:
    out[0,0] += 2*( M2**2 - om_pi*om_kj + np.dot(pvec,kvec) )
    return out / Mtot**2

  elif i==j==1:
    out[0,0] += 2*M2**2 - 0.5*( Ecm2 - E*(om_pi+om_kj) + np.dot(Pvec,pvec+kvec) + om_pi*om_kj - np.dot(pvec,kvec)
                - P2psk_0*kms_0 - P2ksp_0*pms_0 + gam_p*gam_k*(1-np.dot(beta_p_vec,beta_k_vec))*pms_0*kms_0 )
    if waves_ij==('sp','sp'):
      for m in range(-1,2):
        out[m+2,0] += -1/3 * defns.y1real(P2ksp_vec + Vp,m)
        out[0,m+2] += -1/3 * defns.y1real(P2psk_vec + V,m)
      out[1:,1:] += -2/3*t
    return out / Mtot**2

  elif i==1 and j==2:
    out[0,0] += 2*M2**2 - (E-om_pi)*om_kj + np.dot(P2p_vec,kvec) + pms_0*om_ksp
    if waves_ij[0] == 'sp':
      for m in range(-1,2):
        out[m+2,0] += -2/3 * defns.y1real(ksp_vec,m)
    return out / Mtot**2

  elif i==2 and j==1:
    out[0,0] += 2*M2**2 - (E-om_kj)*om_pi + np.dot(P2k_vec,pvec) + kms_0*om_psk
    if waves_ij[1] == 'sp':
      for m in range(-1,2):
        out[0,m+2] += -2/3 * defns.y1real(psk_vec,m)
    return out / Mtot**2
