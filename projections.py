import numpy as np
sqrt=np.sqrt; pi=np.pi; LA=np.linalg
from scipy.linalg import block_diag

import defns, group_theory_defns as GT
import sys

###########################################################
# Here we implement various projections, including the isotropic approx, A1+, and all other irreps
###########################################################
# Pull out l'=l=0 part of full matrix M
def l0_proj(M):
  N = int(len(M)/4)
  M00 = np.zeros((N,N))
  for i in range(N):
    I = 4*i
    for j in range(N):
      J = 4*j
      M00[i][j] = M[I][J]
  return M00

# Pull out l'=l=1 part of full matrix M
def l1_proj(M):
  N = int(len(M)/4)
  M11 = np.zeros((3*N,3*N))
  for i in range(N):
    i_min = 3*i;   i_max = 3*(i+1)
    I_min = 4*i+1; I_max = 4*(i+1)
    for j in range(N):
      j_min = 3*j;   j_max = 3*(j+1)
      J_min = 4*j+1; J_max = 4*(j+1)

      M11[i_min:i_max,j_min:j_max] = M[I_min:I_max,J_min:J_max]
  return M11

################################################################################
# Isotropic approx
################################################################################
# # Matrix projecting l=0 matrix onto A1 for iso approx
# # Note: normalization convention seems irrelevant
# def p_iso00(E,L):
#   # No normalization
#   #  p_iso = np.ones((len(defns.list_nnk(E,L)),1))
#
#   # Steve's convention: each column corresponds to a shell with 1/sqrt(N) normalization (so each vector has length=1)
#   shells = defns.shell_list(E,L)
#   p_iso = np.zeros((len(defns.list_nnk(E,L)),len(shells)))
#   i_k = 0
#   for i_shell in range(len(shells)):
#     N = len(defns.shell_nnk_list(shells[i_shell]))
#     p_iso[i_k:i_k+N,i_shell] = 1/sqrt(N)
#     i_k += N
# #  print(p_iso)
# #  print(np.sum(p_iso,axis=1))
# #  return np.sum(p_iso,axis=1)
#   return p_iso
#
# # Project l'=l=0 matrix into isotropic approx (A1 irrep & l=0)
# def iso_proj00(M00,E,L):
#   p_iso = p_iso00(E,L)
#   return defns.chop( p_iso.T @ M00 @ p_iso )
#
####################################################################################
# 2-pt. projection onto ANY irrep of LG(nnP2) for given nnP2
####################################################################################
# Full projection matrices P^I

# Single 4x4 block (useful for 2-pt. projections)
def P_irrep_2pt(nnP2,I):
  P_block = np.zeros((4,4))
  LG = GT.little_group(nnP2)
  d_I = GT.irrep_dim(I)
  for R in LG:
    P_block += defns.chop( GT.chi(R,I,nnP2) * GT.Dmat(R) )
  return d_I/len(LG)*P_block

def P_irrep_subspace_2pt(nnP2,I):
  P_I = defns.chop(P_irrep_2pt(nnP2,I))
  # eigenvalues should all be 0 or 1; only want 1's
  elist, vlist = LA.eig(P_I)
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace: wrong subspace dimension')
  Psub = defns.chop(vlist[:,ivec].real)
  if ivec==[]:
    return Psub
  else:
    #print(defns.chop(LA.qr(Psub)[0]))
    return defns.chop(LA.qr(Psub)[0])

# Project 4x4 2pt. matrix onto irrep subspace (s+d wave)
def irrep_proj_2pt(M,nnP,I):
  P_I = P_irrep_subspace_2pt(nnP,I)
  return defns.chop( P_I.T @ M @ P_I )



####################################################################################
# 3-pt. projection onto ANY irrep of LG(nnP) for given nnP
####################################################################################
# Full projection matrices (P^I in notes)

# Single shell & single l for given nnP
def P_irrep_o_l(E,L,nnP,I,shell,l,Mijk=[1,1,1],nnk_list=None,waves='sp', parity=-1, lblock=False):
  if nnk_list==None:
    nnk_list = defns.shell_nnk_nnP_list(E,L,nnP,shell, Mijk=Mijk)
  Lk = GT.little_group(shell)
  d_I = GT.irrep_dim(I)
  W = defns.get_lm_size(waves)

  #print(Mijk[0],shell,I,l,W)

  nnP=list(nnP)
  if nnP==[0,0,0]:
    P_shell = []
    for k2 in nnk_list:
      R20 = GT.Rvec(k2,shell)
      P_k2 = []
      for k1 in nnk_list:
        R01 = GT.Rvec(shell,k1)

        P_block = np.zeros((W,W))
        for R in Lk:
          RRR = GT.R_prod(R20,R,R01)
          par = -1 if (parity==-1 and RRR not in GT.rotations_list()) else 1
          if l==0:
            P_block[0,0] += par*GT.chi(RRR,I,nnP)
          elif l==1:
            P_block[1:4,1:4] += par*GT.chi(RRR,I,nnP) * GT.Dmat11(RRR)
            P_block = defns.chop(P_block)
          elif l==2:
            P_block[4:,4:] += par*GT.chi(R,I,nnP) * GT.Dmat22(R)
            P_block = defns.chop(P_block)

        P_k2.append(P_block)
      P_shell.append(P_k2)

    out = d_I/48 * np.block(P_shell)
    if lblock==True:
      if l==0:
        return l0_proj(out)
      elif l==1:
        return l1_proj(out)
    else:
      return out

  else:
    P_shell = []
    LG = GT.little_group(nnP)
    for k2 in nnk_list:
      #R20 = GT.Rvec(k2,shell)
      #print(k2)
      P_k2 = []
      for k1 in nnk_list:
        #R01 = GT.Rvec(shell,k1)

        P_block = np.zeros((W,W))
        for R in LG:
          Rk1 = [np.sign(R[i])*k1[abs(R[i])-1] for i in range(3)]
          #print(Rk1)
          if Rk1 == list(k2):
            #print(Rk1)
            par = -1 if (parity==-1 and R not in GT.rotations_list()) else 1
            if l==0:
              #print(R,I,nnP)
              P_block[0,0] += par*GT.chi(R,I,nnP)
            elif l==1:
              P_block[1:4,1:4] += par*GT.chi(R,I,nnP) * GT.Dmat11(R)
              P_block = defns.chop(P_block)
            elif l==2:
              P_block[4:,4:] += par*GT.chi(R,I,nnP) * GT.Dmat22(R)
              P_block = defns.chop(P_block)


        #print(P_block)
        P_k2.append(P_block)
      #print(P_k2)
      P_shell.append(P_k2)
    #print(shell,P_shell)

    out = d_I/len(LG) * np.block(P_shell)
    #print(out.shape)
    if lblock==True:
      if l==0:
        return l0_proj(out)
      elif l==1:
        return l1_proj(out)
    else:
      return out


# Projection for a given shell
def P_irrep_o(E,L,nnP,I,shell, Mijk=[1,1,1], waves='sp', parity=-1):
  # return P_irrep_o_l(E,L,nnP,I,shell,0) + P_irrep_o_l(E,L,nnP,I,shell,2)    # TB: old code; slower
  nnk_list = defns.shell_nnk_nnP_list(E,L,nnP,shell, Mijk=Mijk)
  Lk = GT.little_group(shell)
  d_I = GT.irrep_dim(I)
  W = defns.get_lm_size(waves)

  nnP=list(nnP)
  if nnP==[0,0,0]:
    P_shell = []
    for k2 in nnk_list:
      R20 = GT.Rvec(k2,shell)
      P_k2 = []
      for k1 in nnk_list:
        R01 = GT.Rvec(shell,k1)
        P_block = np.zeros((W,W))
        for R in Lk:
          RRR = GT.R_prod(R20,R,R01)
          par = -1 if (parity==-1 and RRR not in GT.rotations_list()) else 1
          P_block[0,0] += par*GT.chi(RRR,I,nnP)
          if waves=='sp':
            P_block[1:,1:] += par*GT.chi(RRR,I,nnP) * GT.Dmat11(RRR)
          P_block = defns.chop(P_block)
        P_k2.append(P_block)
      P_shell.append(P_k2)
    out = d_I/48 * np.block(P_shell)
    return out

  else:
    P_shell = []
    LG = GT.little_group(nnP)
    for k2 in nnk_list:
      #R20 = GT.Rvec(k2,shell)
      #print(k2)
      P_k2 = []
      for k1 in nnk_list:
        #R01 = GT.Rvec(shell,k1)

        P_block = np.zeros((W,W))
        for R in LG:
          Rk1 = [np.sign(R[i])*k1[abs(R[i])-1] for i in range(3)]
          #print(Rk1)
          if Rk1 == list(k2):
            #print(Rk1)
            par = -1 if (parity==-1 and R not in GT.rotations_list()) else 1
            P_block[0,0] += par*GT.chi(R,I,nnP)
            if waves=='sp':
              P_block[1:,1:] += par*GT.chi(R,I,nnP) * GT.Dmat11(R)
            P_block = defns.chop(P_block)
        #print(P_block)
        P_k2.append(P_block)
      #print(P_k2)
      P_shell.append(P_k2)
    #print(shell,P_shell)

    out = d_I/len(LG) * np.block(P_shell)
    #print(out.shape)
    return out


# Full projection matrix (includes all shells & l)
def P_irrep_full_i(E,L,nnP,I, Mijk=[1,1,1], waves='sp', parity=-1):
  P_block_list = []
  for shell in defns.shell_list_nnP(E,L,nnP, Mijk=Mijk):
    P_block_list.append( P_irrep_o(E,L,nnP,I,shell, Mijk=Mijk, waves=waves, parity=parity) )
  return block_diag(*P_block_list)

# Full 2+1 projection matrix
def P_irrep_full_2plus1(E,L,nnP,I, M12=[1,1], waves='sp', parity=-1):
  M1, M2 = M12
  P1 = P_irrep_full_i(E,L,nnP,I, Mijk=[M1,M1,M2], waves=waves, parity=parity)
  P2 = P_irrep_full_i(E,L,nnP,I, Mijk=[M2,M1,M1], waves='s', parity=parity)
  return block_diag(*[P1,P2])

# Full ND projection matrix
# def P_irrep_full_ND(E,L,nnP,I, M123=[1,1,1]):
#   P_diag = []
#   for i in range(3):
#     Pi = P_irrep_full_i(E,L,nnP,I, Mijk=defns.get_Mijk(M123,i))
#     P_diag.append(Pi)
#   return block_diag(*P_diag)

############################################
# l'=l=0 part of projection matrix
# def P_irrep_00_2plus1(E,L,nnP,I, M12=[1,1], waves='sp'):
#   return l0_proj(P_irrep_full_2plus1(E,L,nnP,I, M12=M12,waves=waves))

###################################################################
# Subspace projection

# Irrep subspace projection matrix onto specific shell & l (acts on full matrix)
def P_irrep_subspace_o_l(E,L,nnP,I,shell,l,Mijk=[1,1,1],waves='sp', parity=-1, lblock=False):
  P_block_list = []
  for nk in defns.shell_list_nnP(E,L,nnP, Mijk=Mijk):
    if list(nk) == list(shell):
      P_block_list.append( P_irrep_o_l(E,L,nnP,I,shell,l,Mijk=Mijk,waves=waves,parity=parity,lblock=lblock) )
      #print(I,shell,l,np.trace(P_block_list[-1]))
    else:
      P_block_list.append( np.zeros(P_irrep_o_l(E,L,nnP,I,nk,l,Mijk=Mijk,waves=waves,parity=parity,lblock=lblock).shape ))

  P_I = block_diag(*P_block_list)

  elist, vlist = LA.eigh(P_I)

  # eigenvalues should all be 0 or 1; only want 1's
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace: wrong subspace dimension')
  Psub = defns.chop(vlist[:,ivec].real)
  if ivec==[]:
    return Psub
  else:
    #print(defns.chop(LA.qr(Psub)[0]))
    return defns.chop(LA.qr(Psub)[0])


# Irrep projection matrix onto specific shell (acts on full matrix, contains l=0 and l=1)
def P_irrep_subspace_o(E,L,nnP,I,shell, Mijk=[1,1,1], waves='sp', parity=-1):
  P0 = P_irrep_subspace_o_l(E,L,nnP,I,shell,0, Mijk=Mijk, waves=waves, parity=parity)
  #print(I, shell, 'P0:',P0.shape)
  if waves=='s':
    return P0
  elif waves=='sp':
    P1 = P_irrep_subspace_o_l(E,L,nnP,I,shell,1,Mijk=Mijk, waves=waves, parity=parity)
    #print(I, shell, 'P1:',P1.shape)
    return np.concatenate((P0,P1),axis=1)


# Irrep subspace projection matrix (includes all shells & l)
def P_irrep_subspace_i(E,L,nnP,I, Mijk=[1,1,1], waves='sp', parity=-1):
  N = len(defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk))
  W = defns.get_lm_size(waves)
  Psub = np.zeros((W*N,0))
  for shell in defns.shell_list_nnP(E,L,nnP, Mijk=Mijk):
    Psub = np.column_stack((Psub,P_irrep_subspace_o(E,L,nnP,I,shell, Mijk=Mijk, waves=waves, parity=parity)))
  return Psub


def P_irrep_subspace_2plus1(E,L,nnP,I,M12=[1,1], waves='sp', parity=-1):
  M1,M2 = M12
  P1 = P_irrep_subspace_i(E,L,nnP,I, Mijk=[M1,M1,M2], waves=waves, parity=parity)
  P2 = P_irrep_subspace_i(E,L,nnP,I, Mijk=[M2,M1,M1], waves='s', parity=parity)
  return block_diag(*[P1,P2])


# Project matrix onto irrep subspace (all shells & l)
def irrep_proj_2plus1(M,E,L,nnP,I, M12=[1,1], waves='sp', parity=-1):
  P_I = P_irrep_subspace_2plus1(E,L,nnP,I,M12=M12, waves=waves, parity=parity)
  #print(P_I.shape)
  return defns.chop( P_I.T @ M @ P_I )



########################################
# # l'=l=0 irrep subspace projection matrix (acts on F3_00)
# def P_irrep_subspace_00(E,L,nnP,I):
#   N = len(defns.list_nnk_nnP(E,L,nnP))
#   Psub = np.zeros((N,0))
#   for shell in defns.shell_list_nnP(E,L,nnP):
#     Pnew = P_irrep_subspace_o_l(E,L,nnP,I,shell,0,lblock=True)
#     Psub = np.column_stack((Psub,Pnew))
#     #Psub = np.column_stack((Psub,P_irrep_subspace_o_l(E,L,nnP,I,shell,0,lblock=True)))
#   return Psub
#
# # l'=l=2 irrep subspace projection matrix (acts on F3_00)
# def P_irrep_subspace_22(E,L,nnP,I):
#   N = len(defns.list_nnk_nnP(E,L,nnP))
#   Psub = np.zeros((5*N,0))
#   for shell in defns.shell_list_nnP(E,L,nnP):
#     Psub = np.column_stack((Psub,P_irrep_subspace_o_l(E,L,nnP,I,shell,2,lblock=True)))
#   return Psub
#
#
# # Project l'=l=0 matrix onto irrep
# def irrep_proj_00(M00,E,L,nnP,I):
#   Psub = P_irrep_subspace_00(E,L,nnP,I)
#   return defns.chop(Psub.T @ M00 @ Psub)
#
# # Project l'=l=2 matrix onto irrep
# def irrep_proj_22(M22,E,L,nnP,I):
#   Psub = P_irrep_subspace_22(E,L,nnP,I)
#   return defns.chop(Psub.T @ M22 @ Psub)
#
#
####################################################################################
# Decomposition analysis functions
####################################################################################
# Eigenvector decomposition by o,l for given irrep
# def evec_decomp_2plus1(v,E,L,nnP,I,M12=[1,1]):
#   c0_list =[]
#   c1_list = []
#   shells = defns.shell_list_nnP(E,L,nnP)
#   for shell in shells:
#     P0 = P_irrep_subspace_o_l(E,L,nnP,I,shell,0, Mijk=) # TB: need to think through
#     P1 = P_irrep_subspace_o_l(E,L,nnP,I,shell,1)
#
#     c0 = sum([np.dot(v,P0[:,i])**2 for i in range(P0.shape[1])])/LA.norm(v)**2
#     c1 = sum([np.dot(v,P1[:,i])**2 for i in range(P2.shape[1])])/LA.norm(v)**2
#
#     c0_list.append(c0)
#     c1_list.append(c1)
#
#   s = sum(c0_list)+sum(c1_list)
#   print('Eigenvector decomposition for '+str(I)+' (total fraction: '+str(round(s,6))+')')
#   for i in range(len(shells)):
#     if s==0:
#       frac0=0.; frac1=0.
#     else:
#       frac0 = c0_list[i]/s
#       frac1 = c1_list[i]/s
#     print(shells[i],'--- l=0:',round(frac0,8),',\t l=1:',round(frac1,8))
#   print()
#
#   return s
#
#
# # Irrep decomposition of large or small eigenvalues (e.g., poles of F3)
# def pole_decomp(M,E,L,nnP,parity=-1,size='large',thresh=100):
#   out = {}
#   for I in GT.irrep_list(nnP):
#     if parity == -1:
#       I_par = GT.parity_irrep(nnP,I)
#     else:
#       I_par = I
#     #eigs_I = LA.eigvals(irrep_proj(M,E,L,nnP,I)).real
#     P_I = P_irrep_full(E,L,nnP,I)
#     eigs_I = LA.eigvals(P_I@M@P_I).real
#     if size=='large':
#       eigs_I = sorted([e for e in eigs_I if abs(e)>=thresh],key=abs,reverse=True)
#     else:
#       eigs_I = sorted([e for e in eigs_I if abs(e)<=thresh],key=abs)
#     out[I_par] = len(eigs_I)
#     print(I_par+':\t' + str(len(eigs_I)) + '\t' + str(eigs_I))  # comment out if desired
#   print()                                                       # comment out if desired
#   return out
#
# # s-wave version of pole_decomp (e.g. M=F3_00)
# def pole_decomp_00(M,E,L,nnP,parity=-1,size='large',thresh=100):
#   out = {}
#   for I in GT.irrep_list(nnP):
#     if parity == -1:
#       I_par = GT.parity_irrep(nnP,I)
#     eigs_I = LA.eigvals(irrep_proj_00(M,E,L,nnP,I_par)).real
#     if size=='large':
#       eigs_I = [e for e in eigs_I if abs(e)>=thresh]
#     else:
#       eigs_I = [e for e in eigs_I if abs(e)<=thresh]
#     out[I] = eigs_I #len(eigs_I)
#   #  print(I+':\t' + str(len(eigs_I)) + '\t' + str(eigs_I)) # comment out if desired
#   #print()                                                  # comment out if desired
#   return out
#
#
#####################################################################
# Graveyard (old code)
#####################################################################
# # These functions aren't useful since F3 connects different shells & \ell's
#
# # Project matrix onto irrep subspace for single shell & l
# def irrep_proj_o_l(M,E,L,I,shell,l):
#   P_I = P_irrep_subspace_o_l(E,L,I,shell,l)
#   return defns.chop( P_I.T @ M @ P_I )
#
# # Project matrix onto irrep subspace for single shell (contains l=0 and l=2)
# def irrep_proj_o(M,E,L,I,shell):
#   P_I = P_irrep_subspace_o(E,L,I,shell)
#   return defns.chop( P_I.T @ M @ P_I )
