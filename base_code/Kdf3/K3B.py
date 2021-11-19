import defns
import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg

#################################################################
# Want to compute K3B term in Kdf3 for 2+1 systems
#################################################################

# Compute contributions to K3B for given (i,pvec; j,kvec)
def K3B_ij_pk(E,Pvec,pvec,kvec, i,j, M12=[1,1], waves_ij=('sp','sp')):
  M1,M2 = M12; Mtot = 2*M1+M2
  Mi,Mj = [M12[i-1],M12[j-1]]

  Ecm2 = E**2 - sum([x**2 for x in Pvec])
  P2p_vec = Pvec-pvec; P2k_vec = Pvec-kvec

  om_pi = sqrt(sum([x**2 for x in pvec])+Mi**2)
  om_kj = sqrt(sum([x**2 for x in kvec])+Mj**2)

  if i==1:
    q2_p1 = defns.qst2_i(E,Pvec,pvec, Mijk=[M1,M1,M2])
    om_qp1_1 = sqrt(q2_p1 + M1**2); om_qp1_2 = sqrt(q2_p1 + M2**2)
    pms_0 = om_qp1_1 - om_qp1_2
    pp_vec1 = defns.boost(om_pi,pvec,E-om_pi,P2p_vec)
    om_pp1 = sqrt(sum([x**2 for x in pp_vec1]) + M1**2)
  if j==1:
    q2_k1 = defns.qst2_i(E,Pvec,kvec, Mijk=[M1,M1,M2])
    om_qk1_1 = sqrt(q2_k1 + M1**2); om_qk1_2 = sqrt(q2_k1 + M2**2)
    kms_0 = om_qk1_1 - om_qk1_2
    kk_vec1 = defns.boost(om_kj,kvec,E-om_kj,P2k_vec)
    om_kk1 = sqrt(sum([x**2 for x in kk_vec1]) + M1**2)


  Wi = defns.get_lm_size(waves_ij[0])
  Wj = defns.get_lm_size(waves_ij[1])
  out = np.zeros((Wi,Wj))

  if i==j==2:
    out[0,0] += 2*( Ecm2 - E*(om_pi+om_kj) + np.dot(Pvec,pvec+kvec) + M2**2 - 4*M1**2 )
    return out / Mtot**2

  elif i==j==1:
    out[0,0] += E*(om_pi+om_kj) - np.dot(Pvec,pvec+kvec) - 6*M1**2 + om_pp1*pms_0 + om_kk1*kms_0
    if waves_ij==('sp','sp'):
      for m in range(-1,2):
        out[m+2,0] += -2/3*defns.y1real(pp_vec1,m)  # TB: no qp
        out[0,m+2] += -2/3*defns.y1real(kk_vec1,m)  # TB: no qk
    return out / Mtot**2

  elif i==1 and j==2:
    out[0,0] += E*om_pi - np.dot(pvec,Pvec) + Ecm2 - 2*E*om_kj + 2*np.dot(Pvec,kvec) + om_pp1*pms_0 + M2**2 - 7*M1**2
    if waves_ij[0] == 'sp':
      for m in range(-1,2):
        out[m+2,0] += -2/3*defns.y1real(pp_vec1,m)  # TB: no qp
    return out / Mtot**2

  elif i==2 and j==1:
    out[0,0] += E*om_kj - np.dot(kvec,Pvec) + Ecm2 - 2*E*om_pi + 2*np.dot(Pvec,pvec) + om_kk1*kms_0 + M2**2 - 7*M1**2
    if waves_ij[1] == 'sp':
      for m in range(-1,2):
        out[0,m+2] += -2/3*defns.y1real(kk_vec1,m)  # TB: no qk
    return out / Mtot**2
