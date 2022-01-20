import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
from scipy.linalg import block_diag

import defns, group_theory_defns as GT
import sys

################################################################################
''' Here we implement projections onto all little group irreps '''
################################################################################
# 3-pt. projection onto ANY irrep of LG(nnP) for given nnP
################################################################################
# Full projection matrices (P_I in notes)
# Single orbit & single l for given nnP
def P_irrep_o_l(nnP,I,orbit,l,Mijk=[1,1,1], parity=-1):
  d_I = GT.irrep_dim(I)
  W = 2*l+1
  LG = GT.little_group(nnP)
  nnk_list = defns.orbit_nnk_list(orbit,nnP)

  P_ol = []
  for k2 in nnk_list:
    P_k2 = []
    for k1 in nnk_list:
      P_block = np.zeros((W,W))
      for R in LG:
        Rk1 = GT.cubic_transf(k1,R)
        if Rk1 == list(k2):
          par = -1 if (parity==-1 and R not in GT.rotations_list()) else 1
          if l==0:
            D = 1
          elif l==1:
            D = GT.Dmat11(R)
          elif l==2:
            D = GT.Dmat22(R)
          P_block += par*GT.chi(R,I,nnP) * D
      #print(P_block)
      P_k2.append(defns.chop(P_block))
    P_ol.append(P_k2)
  out = d_I/len(LG) * np.block(P_ol)
  return out

# Projection for a given orbit
def P_irrep_o(nnP,I,orbit, Mijk=[1,1,1], waves='sp', parity=-1):
  if waves=='s':
    return P_irrep_o_l(nnP,I,orbit,0, Mijk=Mijk, parity=parity)
  # elif waves=='sp':
  #   P0 = P_irrep_o_l(nnP,I,orbit,0, Mijk=Mijk, parity=parity)
  #   P1 = P_irrep_o_l(nnP,I,orbit,1, Mijk=Mijk, parity=parity)
  #   return block_diag(*[P0,P1])   # less efficient than below code
  d_I = GT.irrep_dim(I)
  W = defns.get_lm_size(waves)
  LG = GT.little_group(nnP)
  nnk_list = defns.orbit_nnk_list(orbit,nnP)
  P_o = []
  for k2 in nnk_list:
    P_k2 = []
    for k1 in nnk_list:
      P_block = np.zeros((W,W))
      for R in LG:
        Rk1 = GT.cubic_transf(k1,R)
        if Rk1 == list(k2):
          par = -1 if (parity==-1 and R not in GT.rotations_list()) else 1
          P_block[0,0] += par*GT.chi(R,I,nnP)
          if waves=='sp':
            P_block[1:,1:] += par*GT.chi(R,I,nnP) * GT.Dmat11(R)
          P_block = defns.chop(P_block)
      #print(P_block)
      P_k2.append(P_block)
    P_o.append(P_k2)
  out = d_I/len(LG) * np.block(P_o)
  return out

# Full flavor-i projection block (includes all orbits & waves)
def P_irrep_full_i(nnP,I, orbit_list, Mijk=[1,1,1], waves='sp', parity=-1):
  P_block_list = []
  for orbit in orbit_list:
    P_block_list.append( P_irrep_o(nnP,I,orbit, Mijk=Mijk, waves=waves, parity=parity) )
  return block_diag(*P_block_list)

# Full 2+1 projection matrix
def P_irrep_full_2plus1(nnP,I, orbit_lists_12, M12=[1,1], waves='sp', parity=-1):
  M1, M2 = M12
  P1 = P_irrep_full_i(nnP,I, orbit_lists_12[0], Mijk=[M1,M1,M2], waves=waves, parity=parity)
  P2 = P_irrep_full_i(nnP,I, orbit_lists_12[1], Mijk=[M2,M1,M1], waves='s', parity=parity)
  return block_diag(*[P1,P2])

################################################################################
# Full ND projection matrix (s-wave only)
def P_irrep_full_ND(nnP,I, orbit_lists_123, M123=[1,1,1], parity=-1):
  P_diag = []
  for i in range(3):
    Pi = P_irrep_full_i(nnP,I, orbit_lists_123[i], Mijk=defns.get_Mijk(M123,i), waves='s', parity=parity)
    P_diag.append(Pi)
  return block_diag(*P_diag)
################################################################################
# Full ID projection matrix (s-wave only)
def P_irrep_full_ID(nnP,I, orbit_list, parity=-1):
  return P_irrep_full_i(nnP,I, orbit_list, Mijk=[1,1,1], waves='s', parity=parity)
################################################################################

################################################################################
# Subspace projection
################################################################################
# Irrep subspace projection matrix sub-block for specific shell & l (size 2l+1 x 2l+1)
def P_irrep_subspace_o_l(nnP,I,orbit,l,Mijk=[1,1,1],parity=-1):
  P_I = P_irrep_o_l(nnP,I,orbit,l,Mijk=Mijk,parity=parity)
  elist, vlist = LA.eigh(P_I)
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]    # eigenvalues should all be 0 or 1; only want 1's
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace_o_l: wrong subspace dimension')
    raise ValueError
  Psub = defns.chop( vlist[:,ivec].real )
  if ivec!=[]:
    Psub = defns.chop( LA.qr(Psub)[0] )
  return Psub


# Irrep projection matrix block for specific orbit (contains all waves, size WxW)
def P_irrep_subspace_o(nnP,I,orbit, Mijk=[1,1,1], waves='sp', parity=-1):
  P_I = P_irrep_o(nnP,I,orbit, Mijk=Mijk, waves=waves, parity=parity)
  elist, vlist = LA.eigh(P_I)
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]    # eigenvalues should all be 0 or 1; only want 1's
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace_o: wrong subspace dimension')
    raise ValueError
  Psub = defns.chop( vlist[:,ivec].real )
  if ivec!=[]:
    Psub = defns.chop( LA.qr(Psub)[0] )
  return Psub


# Irrep subspace flavor-i projection block (includes all orbits & waves)
def P_irrep_subspace_i(nnP,I,orbit_list, Mijk=[1,1,1], waves='sp', parity=-1):
  P_o_list = []
  for orbit in orbit_list:
    P_o = P_irrep_subspace_o(nnP,I,orbit, Mijk=Mijk, waves=waves, parity=parity)
    #print(Psub.shape,P_o.shape)
    P_o_list.append(P_o)
  Psub = block_diag(*P_o_list)
  return Psub

# 2+1 irrep subspace projection matrix (includes all flavors, orbits, & waves)
def P_irrep_subspace_2plus1(nnP,I,orbit_lists_12,M12=[1,1], waves='sp', parity=-1):
  M1,M2 = M12
  P1 = P_irrep_subspace_i(nnP,I, orbit_lists_12[0], Mijk=[M1,M1,M2], waves=waves, parity=parity)
  P2 = P_irrep_subspace_i(nnP,I, orbit_lists_12[1], Mijk=[M2,M1,M1], waves='s', parity=parity)
  return block_diag(*[P1,P2])

# Project 2+1 matrix onto irrep subspace (all flavors, orbits, & waves)
def irrep_proj_2plus1(M,E,L,nnP,I, orbit_lists_12=None, M12=[1,1], waves='sp', parity=-1):
  M1,M2 = M12
  if orbit_lists_12==None:
    orbit_lists_12 = [defns.orbit_list_nnP(E,L,nnP, Mijk=[M1,M1,M2]), defns.orbit_list_nnP(E,L,nnP, Mijk=[M2,M1,M1])]
  P_I = P_irrep_subspace_2plus1(nnP,I,orbit_lists_12,M12=M12, waves=waves, parity=parity)
  #print(P_I.shape)
  return defns.chop( P_I.T @ M @ P_I )


################################################################################
# ND irrep subspace projection matrix (includes all flavors & orbits, s-wave only)
def P_irrep_subspace_ND(nnP,I,orbit_lists_123,M123=[1,1,1], parity=-1):
  Pi_list = []
  for i in range(3):
    Pi = P_irrep_subspace_i(nnP,I, orbit_lists_123[i], Mijk=defns.get_Mijk(M123,i), waves='s', parity=parity)
    Pi_list.append(Pi)
  return block_diag(*Pi_list)


# Project ND matrix onto irrep subspace (all orbits, s-wave only)
def irrep_proj_ND(M,E,L,nnP,I, orbit_lists_123=None, M123=[1,1,1], parity=-1):
  #M1,M2,M3 = M123
  if orbit_lists_123==None:
    orbit_lists_123 = [defns.orbit_list_nnP(E,L,nnP, Mijk=defns.get_Mijk(M123,i)) for i in range(3)]
  P_I = P_irrep_subspace_ND(nnP,I,orbit_lists_123,M123=M123, parity=parity)
  #print(P_I.shape)
  return defns.chop( P_I.T @ M @ P_I )

################################################################################
# ID irrep subspace projection matrix (includes all orbits, s-wave only)
def P_irrep_subspace_ID(nnP,I,orbit_list, parity=-1):
  return P_irrep_subspace_i(nnP,I,orbit_list, Mijk=[1,1,1], waves='s', parity=parity)


# Project ID matrix onto irrep subspace (all orbits, s-wave only)
def irrep_proj_ID(M,E,L,nnP,I, orbit_list=None,parity=-1):
  if orbit_list==None:
    orbit_list = defns.orbit_list_nnP(E,L,nnP, Mijk=[1,1,1])
  P_I = P_irrep_subspace_ID(nnP,I,orbit_list, parity=parity)
  #print(P_I.shape)
  return defns.chop( P_I.T @ M @ P_I )


################################################################################
# 2-pt. projection onto ANY irrep of LG(nnP2) for given nnP2
################################################################################
# Full projection matrices P^I
def P_irrep_2pt(nnP2,I,waves='sp'):
  LG = GT.little_group(nnP2)
  d_I = GT.irrep_dim(I)
  W = defns.get_lm_size(waves)
  P_block = np.zeros((W,W))
  for R in LG:
    if 's' in waves:
      P_block[0,0] += GT.chi(R,I,nnP2)
    if waves=='sp':
      #print(R,GT.chi(R,I,nnP2), GT.Dmat11(R),'\n')
      P_block[1:,1:] += defns.chop( GT.chi(R,I,nnP2) * GT.Dmat11(R) )
  return d_I/len(LG)*P_block

# Subspace matrices
def P_irrep_subspace_2pt(nnP2,I,waves='sp'):
  P_I = defns.chop(P_irrep_2pt(nnP2,I,waves=waves))
  # eigenvalues should all be 0 or 1; only want 1's
  elist, vlist = LA.eig(P_I)
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace_2pt: subspace dimension mismatch')
    raise ValueError
  Psub = defns.chop(vlist[:,ivec].real)
  if ivec!=[]:
    Psub = defns.chop( LA.qr(Psub)[0] )
  return Psub

# Project 2-pt. matrix onto irrep subspace
def irrep_proj_2pt(M,nnP,I,waves='sp'):
  P_I = P_irrep_subspace_2pt(nnP,I,waves=waves)
  return defns.chop( P_I.T @ M @ P_I )

################################################################################
# Graveyard (old code)
################################################################################
# if waves=='s':
#   return P_irrep_subspace_o_l(nnP,I,orbit, 0, Mijk=Mijk,parity=parity)
# elif waves=='sp':
#   P0 = P_irrep_subspace_o_l(nnP,I,orbit, 0, Mijk=Mijk,parity=parity)
#   P1 = P_irrep_subspace_o_l(nnP,I,orbit, 1, Mijk=Mijk,parity=parity)
#   return block_diag(*[P0,P1])   # less efficient than above code
