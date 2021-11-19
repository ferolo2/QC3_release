import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg;

import K3B, K3E
import defns

#################################################################
# Full linear-order threshold expansion of Kdf3 for 2+1 systems
#################################################################

# Note: input order for all functions is (E,Pvec,outgoing momenta,incoming momenta)

# Compute 4x4 block of K3 for given (i,pvec; j,kvec)
def K3_ij_pk(E,Pvec,pvec,kvec, i,j, Kiso, K3B_par,K3E_par, M12=[1,1], waves_ij=('sp','sp')):
  M1,M2 = M12; Mtot = 2*M1+M2
  Ecm2 = E**2-sum([x**2 for x in Pvec])
  d = Ecm2/Mtot**2 - 1

  Wi = defns.get_lm_size(waves_ij[0])
  Wj = defns.get_lm_size(waves_ij[1])
  out = np.zeros((Wi,Wj))

  out[0,0] += sum([Kiso[ii]*d**ii for ii in range(len(Kiso))])
  if K3B_par!=0:
    out += K3B_par*K3B.K3B_ij_pk(E,Pvec,pvec,kvec, i,j, M12=M12, waves_ij=waves_ij)
  if K3E_par!=0:
    out += K3E_par*K3E.K3E_ij_pk(E,Pvec,pvec,kvec, i,j, M12=M12, waves_ij=waves_ij)

  # if out.imag>1e-15:
  #   print('Error in K3quad: imaginary part in real basis output')
  return out


# Full ij block
def K3mat_ij(E,L,nnP, i,j, Kiso,K3B_par,K3E_par, M12=[1,1], waves_ij=('sp','sp'), nnk_lists_12=None):
  M1,M2=M12
  if nnk_lists_12==None:
    nnp_list = defns.list_nnk_nnP(E,L,nnP, Mijk=defns.get_Mijk([M1,M1,M2],i))
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=defns.get_Mijk([M1,M1,M2],j))
  else:
    (nnp_list, nnk_list) = (nnk_lists_12[i-1], nnk_lists_12[j-1])
  Pvec = [x*2*pi/L for x in nnP]
  Ni = len(nnp_list); Nj = len(nnk_list)
  Wi = defns.get_lm_size(waves_ij[0])
  Wj = defns.get_lm_size(waves_ij[1])
  K3full = np.zeros((Ni*Wi,Nj*Wj))
  for p,nnp in enumerate(nnp_list):
    pvec = np.array([x*2*pi/L for x in nnp])
    for k,nnk in enumerate(nnk_list):
      kvec = np.array([x*2*pi/L for x in nnk])
      K3pk = K3_ij_pk(E,Pvec,pvec,kvec, i,j, Kiso, K3B_par,K3E_par, M12=M12, waves_ij=waves_ij)
      K3full[Wi*p:Wi*(p+1),Wj*k:Wj*(k+1)] = K3pk
      #print((i,j), (nnp,nnk), '\n', K3pk)
  return K3full


# Full matrix
def K3mat_2plus1(E,L,nnP, Kiso,K3B_par,K3E_par, M12=[1,1], waves='sp', nnk_lists_12=None):
  M1,M2 = M12
  if nnk_lists_12==None:
    nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
    nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
    nnk_lists_12 = [nnk_list_1, nnk_list_2]


  K11 = K3mat_ij(E,L,nnP, 1,1, Kiso,K3B_par,K3E_par, M12=M12, waves_ij=(waves,waves), nnk_lists_12=nnk_lists_12)
  K12 = K3mat_ij(E,L,nnP, 1,2, Kiso,K3B_par,K3E_par, M12=M12, waves_ij=(waves,'s'), nnk_lists_12=nnk_lists_12)
  K21 = K12.T #K3mat_ij(E,L,nnP, 2,1, Kiso,K3B_par,K3E_par, M12=M12, waves_ij=('s',waves), nnk_lists_12=nnk_lists_12)
  K22 = K3mat_ij(E,L,nnP, 2,2, Kiso,K3B_par,K3E_par, M12=M12, waves_ij=('s','s'), nnk_lists_12=nnk_lists_12)

  #print(np.amax(abs(K12-K21.T)))
  K3full = [[K11, K12/sqrt(2)], [K21/sqrt(2), K22/2]]
  #print(K22)
  #print(K11.shape,K12.shape,K21.shape,K22.shape)
  return np.block(K3full)


################################################################################
# ND s-wave case (isotropic terms only)
################################################################################
# Full matrix
def K3mat_ND_iso(E,L,nnP, Kiso, M123=[1,1,1], nnk_lists_123=None):
  M1,M2,M3 = M123
  Mtot = sum(M123)
  Pvec = 2*pi/L * np.array(nnP)
  Ecm2 = E**2-sum([x**2 for x in Pvec])
  d = Ecm2/Mtot**2 - 1
  val = sum([Kiso[ii]*d**ii for ii in range(len(Kiso))])

  if nnk_lists_123==None:
    nnk_lists_123 = []
    for i in range(3):
      Mijk = defns.get_Mijk(M123,i)
      nnk_lists_123.append(defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk))

  N123 = sum([len(x) for x in nnk_lists_123])
  K3full = val * np.ones((N123,N123))
  return K3full

################################################################################
# ID s-wave case (isotropic terms only)
################################################################################
# Full matrix
def K3mat_ID_iso(E,L,nnP, Kiso, nnk_list=None):
  Pvec = 2*pi/L * np.array(nnP)
  Ecm2 = E**2-sum([x**2 for x in Pvec])
  d = Ecm2/9 - 1
  val = sum([Kiso[ii]*d**ii for ii in range(len(Kiso))])
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])
  N = len(nnk_list)
  K3full = val * np.ones((N,N))
  return K3full
