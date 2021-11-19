import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
# import sums_mov as sums
import defns
# from numba import jit,njit

################################################################################
# Compute individual matrix element of Gtilde^{ij} = G^{ij}/(2*omega*L^3)
################################################################################
#@njit(fastmath=True,cache=True)
def G_ij(E, L, nnp, nnk,l1,m1,l2,m2,nnP, Mijk=[1,1,1], return_all=False):
    # Setting return_all = True returns [G, nnks, nnps]; just G otherwise
    [Mpi, Mkj, Mbk] = Mijk

    twopibyL = 2*pi/L
    p = LA.norm(nnp) * twopibyL
    k = LA.norm(nnk) * twopibyL
    pk = LA.norm(nnk+nnp) * twopibyL
    nPk = LA.norm(nnk-nnP)
    Pk = nPk*twopibyL
    nPp = LA.norm(nnp-nnP)
    Pp = nPp*twopibyL

    omp = sqrt(Mpi**2+p**2)
    omk = sqrt(Mkj**2+k**2)
    #ompk = np.sqrt(1+pk**2)

    pvec = nnp*twopibyL
    kvec = nnk*twopibyL
    Pvec = nnP*twopibyL
    bvec = Pvec-pvec-kvec

    bkp2 = (E-omp-omk)**2 - sum(bvec**2)

    sig_pi = defns.sigma_i(E,Pvec,pvec,Mi=Mpi)
    sig_kj = defns.sigma_i(E,Pvec,kvec,Mi=Mkj)

    out = defns.hh(sig_pi,Mjk=[Mkj,Mbk])*defns.hh(sig_kj,Mjk=[Mpi,Mbk]) / (L**6 * 4*omp*omk*(bkp2-Mbk**2))
    #out = defns.hh(E2p2)*defns.hh(E2k2)/(L**3 * 4*omp*omk*(bkp2-1))

    # nnks and nnps are the full vectors k* and p*
    nnks = defns.boost(omk, kvec, E-omp, Pvec-pvec)
    nnps = defns.boost(omp, pvec, E-omk, Pvec-kvec)

    out *= defns.ylm(nnks,l1,m1) * defns.ylm(nnps,l2,m2)

    if return_all==True:
      return out.real, nnks, nnps
    return out.real

################################################################################
# Compute block matrix Gtilde^{ij} = G^{ij}/(2*omega*L^3)
################################################################################
def Gmat_ij(E,L,nnP, Mijk=[1,1,1], nnp_list=None, nnk_list=None, waves_ij=('sp','sp')):
  [Mi,Mj,Mk] = Mijk
  if nnp_list==None:
    nnp_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[Mj,Mi,Mk])
  # Np = len(nnp_list)
  # Nk = len(nnk_list)
  #nnk_list = max([nnk_list,nnp_list], key=len) # need square matrix (pad G with zeros)
  #N = len(nnk_list)

  Wi = defns.get_lm_size(waves_ij[0])
  Wj = defns.get_lm_size(waves_ij[1])
  #print(nnk_list)
  #print(N)
  Gfull = []
  for nnp in nnp_list:
    Gp = []
    for nnk in nnk_list:
      Gpk_00, nnks, nnps = G_ij(E,L,np.array(nnp),np.array(nnk),0,0,0,0,np.array(nnP), Mijk=Mijk, return_all=True)
      Gpk = Gpk_00 * np.ones((Wi,Wj))
      # Multiply by spherical harmonics
      for i1 in range(1,Wi):
        l1, m1 = defns.lm_idx(i1,waves=waves_ij[0])
        Gpk[i1,:] *= defns.ylm(nnks,l1,m1)
      for i2 in range(1,Wj):
        l2, m2 = defns.lm_idx(i2,waves=waves_ij[1])
        Gpk[:,i2] *= defns.ylm(nnps,l2,m2)
      Gp.append(Gpk)
    Gfull.append(Gp)
  return defns.chop(np.block(Gfull))

################################################################################
# Compute parity matrix PL of size (Ntot,Ntot)
################################################################################
def get_parity_block(Ntot,waves='spd'):
  PL = np.eye(Ntot)
  for i in range(Ntot):
    l = defns.lm_idx(i,waves=waves)[0]
    if l%2 == 1:
      PL[i,i] = -1
  return PL

################################################################################
# Full 2+1 matrix made of blocks Gtilde^{ij} = G^{ij}/(2*omega*L^3)
################################################################################
def Gmat_2plus1(E,L,nnP, M12=[1,1], nnk_lists_12=None, waves='sp'):
  M1, M2 = M12
  M123 = [M1, M1, M2]
  # Make lists of spectator momenta (if not input)
  if nnk_lists_12 == None:
    nnk_lists_12 = []
    for i in range(2):
      Mijk = defns.get_Mijk(M12,i)
      nnk_list_i = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
      nnk_lists_12.append(nnk_list_i)

  W = defns.get_lm_size(waves)
  N1_tot = len(nnk_lists_12[0]) * W
  N2_tot = len(nnk_lists_12[1]) #* W

  # Make parity block PL_1
  PL_1 = get_parity_block(N1_tot,waves=waves)

  # Now construct full 2+1 G-matrix
  G11 = Gmat_ij(E,L,nnP, Mijk=[M1,M1,M2], nnp_list=nnk_lists_12[0], nnk_list=nnk_lists_12[0], waves_ij=(waves,waves))
  G12 = sqrt(2) * PL_1 @ Gmat_ij(E,L,nnP, Mijk=[M1,M2,M1], nnp_list=nnk_lists_12[0], nnk_list=nnk_lists_12[1], waves_ij=(waves,'s'))
  G21 = G12.T
  G22 = np.zeros((N2_tot,N2_tot))
  Gfull = [[G11, G12],[G21, G22]]
  return defns.chop(np.block(Gfull))


################################################################################
# Full ND matrix made of blocks Gtilde^{ij} = G^{ij}/(2*omega*L^3); s-wave only
################################################################################
def Gmat_ND(E,L,nnP, M123=[1,1,1], nnk_lists_123=None):
  waves = 's' # only use s-wave for ND case (Kdf3 isn't ready for p-wave)
  # Make lists of spectator momenta (if not input)
  if nnk_lists_123 == None:
    nnk_lists_123 = []
    for i in range(3):
      Mijk = defns.get_Mijk(M123,i)
      nnk_list_i = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
      nnk_lists_123.append(nnk_list_i)

  W = defns.get_lm_size(waves)
  # Make list of parity blocks PL_i
  PL_list = []
  for i in range(3):
    Ni_tot = len(nnk_lists_123[i]) * W
    PL_list.append(get_parity_block(Ni_tot,waves=waves))

  # Now construct full ND G-matrix
  Gfull = []
  for i in range(3):
    nnp_list = nnk_lists_123[i]
    Ni_tot = len(nnp_list)*W
    Gi_row = []
    for j in range(3):
      if j==i:
        Gij_block = np.zeros((Ni_tot,Ni_tot))
      elif j>i:
        Mijk = defns.get_Mijk(M123,i,j=j)
        nnk_list = nnk_lists_123[j]
        Gij_block = Gmat_ij(E,L,nnP, Mijk=Mijk, nnp_list=nnp_list, nnk_list=nnk_list, waves_ij=(waves,waves))
        # Apply parity factor
        if j==(i+1)%3:
          Gij_block = Gij_block @ PL_list[j]
        elif j==(i-1)%3:
          Gij_block = PL_list[i] @ Gij_block
      elif j<i:
        Gij_block = Gfull[j][i].T

      Gi_row.append(Gij_block)
    Gfull.append(Gi_row)
  return defns.chop(np.block(Gfull))



################################################################################
# Full ID (identical particles) matrix Gtilde = G/(2*omega*L^3)
################################################################################
def Gmat_ID(E,L,nnP, nnk_list=None):
  waves = 's'   # only use s-wave for ID case (Kdf3 isn't ready for d-wave)
  # Make lists of spectator momenta (if not input)
  if nnk_list == None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])

  W = defns.get_lm_size(waves)
  N_tot = len(nnk_list) * W

  Gfull = Gmat_ij(E,L,nnP, Mijk=[1,1,1], nnp_list=nnk_list, nnk_list=nnk_list, waves_ij=(waves,waves))
  return defns.chop(Gfull)
