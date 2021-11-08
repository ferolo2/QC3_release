import numpy as np, sys
pi=np.pi; LA=np.linalg; exp=np.exp
from scipy.linalg import block_diag
from scipy.special import erfi,erfc
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import defns
from constants import *
sqrt = defns.sqrt

################################################################################
# Find maximum n needed in UV regime for sum_smooth_nnk & sum_full_nnk
################################################################################
def getnmaxreal(cutoff,hhk,gam,x2):
  # TB: added hhk*gam factor to match beyond_iso paper
  # Note: hhk factor helps avoid runtime issue at shell thresholds (large gamma, but tiny hhk)
  alpha = get_alpha()
  f = lambda lam: hhk*gam * 2*pi*sqrt(pi/alpha)*exp(alpha*x2) * erfc(sqrt(alpha)*lam) - cutoff
  lam = fsolve(f,5)[0]
  nmax_real = lam*gam
  # print('hhk*gam, nmax_real:',hhk*gam,nmax_real)
  return nmax_real
################################################################################
# Compute summand, modulo a common overall constant factor
################################################################################
def summand(x2, rvec, l1,m1,l2,m2):
  r2 = sum(rvec**2)
  prop_den = x2-r2
  # const = hhk/(16*pi**2*L**4*omk*E2k)   # put in at end
  UV_term = exp(get_alpha()*prop_den)
  Ylm_term = defns.ylm(rvec,l1,m1) * defns.ylm(rvec,l2,m2) # TB: no q (put 2pi/L in separately)
  return UV_term * Ylm_term * 1/prop_den  #*const
################################################################################
# Compute FULL sum w/o splitting pole/smooth parts
################################################################################
def sum_full_nnk(E,nnP,L,nnk, Mijk=[1,1,1], waves='sp'):
  [Mi, Mj, Mk] = Mijk
  W = defns.get_lm_size(waves)
  twopibyL = 2*pi/L

  kvec = np.array(nnk)*twopibyL
  omk = sqrt(sum(kvec**2)+Mi**2)
  E2k = E-omk
  nnP2k = nnP-nnk
  sig_i = E2k**2 - sum(nnP2k**2)*twopibyL**2
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  alpha_ij = sqrt(q2_i + Mj**2) / sqrt(sig_i)

  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])
  if hhk==0:
    return np.zeros((W,W))

  gam = E2k/sqrt(sig_i)
  x2 = q2_i/twopibyL**2

  cutoff = get_cutoff()
  nmax_real = getnmaxreal(cutoff,hhk,gam,x2)
  nmax = int(np.ceil(nmax_real))

  #nna_on_list = list_nna_on(L,nnP,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)

  out = np.zeros((W,W))
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nna = np.array([n1,n2,n3])
        if LA.norm(nna)<nmax_real: # and list(nna) not in nna_on_list:
          if list(nnP2k)==[0,0,0]:
            rvec = nna
          else:
            rvec = nna + nnP2k * (np.dot(nna,nnP2k)/sum(nnP2k**2) * (1/gam-1) - alpha_ij/gam)
          for i1 in range(W):
            [l1,m1] = defns.lm_idx(i1)
            for i2 in range(W):
              [l2,m2] = defns.lm_idx(i2)
              out[i1,i2] += (2*pi/L)**(l1+l2) * summand(x2,rvec, l1,m1,l2,m2)
  return out # *const


################################################################################
# Compute PV integral
################################################################################
def int_nnk(L,gam,x2,waves='sp'):
  alpha = get_alpha()
  W = defns.get_lm_size(waves)

  twopibyL = 2*pi/L
  x = sqrt(x2)
  ax2 = alpha*x2
  e_ax2 = exp(ax2)
  erfi_sqrt_ax2 = erfi(sqrt(ax2))
  c = 4*pi*gam

  out_dict = {}
  if 's' in waves:
    factor1 = -sqrt(pi/alpha)*0.5*e_ax2
    factor2 = 0.5*pi*x*erfi_sqrt_ax2
    out_dict[0] = c*(factor1 + factor2)
  if 'p' in waves:
    factor1 = -sqrt(pi/alpha**3)*(1+2*ax2)*e_ax2/4
    factor2 = 0.5*pi*x**3*erfi_sqrt_ax2
    out_dict[1] = twopibyL**2 * c * (factor1 + factor2) # no q factors
  if 'd' in waves:
    factor1 = -sqrt(pi/alpha**5)*(3+2*ax2+4*ax2**2)*e_ax2/8
    factor2 = 0.5*pi*x**5*erfi_sqrt_ax2
    out_dict[2] = twopibyL**4 * c * (factor1 + factor2) # no q factors

  out = np.zeros((W,W))
  for i in range(W):
    l, _ = defns.lm_idx(i,waves=waves)
    if abs(out_dict[l].imag)>1e-13:
      sys.error('Error in F_fast: imaginary part encountered')
      print(out_dict[l])
      raise ValueError
    out[i,i] = out_dict[l].real
  return out

################################################################################
# Full \wt{F}^{(i)}(k) matrix w/o splitting pole/smooth terms
################################################################################
def F_i_nnk(E,nnP,L,nnk, Mijk=[1,1,1], waves='sp'):
  [Mi,Mj,Mk] = Mijk

  twopibyL = 2*pi/L
  Pvec = np.array(nnP)*twopibyL
  kvec = np.array(nnk)*twopibyL
  omk = sqrt(sum(kvec**2)+Mi**2)
  E2k = E - omk

  sig_i = defns.sigma_i(E,Pvec,kvec,Mi=Mi)
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])

  gam = E2k/sqrt(sig_i)
  x2 = q2_i/twopibyL**2


  # const = 1/(32*pi**2*L*omk*E2k)
  const = hhk/(16*pi**2*L**4*omk*E2k) # TB: assumes no flavor symmetry; divide by 2 for symmetric case
                                      # TB: convention is L for beyond_iso, L**4 for TOPT papers
  sum_bare = sum_full_nnk(E,nnP,L,nnk, Mijk=Mijk, waves=waves)
  int_bare = int_nnk(L,gam,x2,waves=waves)
  # print(sum_bare)
  # print(int_bare)
  return const*(sum_bare-int_bare)

################################################################################
# Full \wt{F}^{(i)} matrix computed from scratch
################################################################################
def F_i_full_scratch(E,nnP,L, Mijk=[1,1,1], waves='sp', nnk_list=None, diag_only=True):
  if nnk_list==None:
    nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
  Fi_diag = []
  for nnk in nnk_list:
    Fi_diag.append(F_i_nnk(E,nnP,L,nnk, Mijk=Mijk, waves=waves))
  if diag_only==True:
    return Fi_diag
  else:
    return block_diag(*Fi_diag)

################################################################################
# Full 2+1 matrix \wt{F} computed from scratch
################################################################################
def F_full_2plus1_scratch(E,nnP,L, M12=[1,1], waves='sp', nnk_lists_12=None, diag_only=True):
  M1, M2 = M12
  #M123 = [M1, M1, M2]
  if nnk_lists_12 == None:
    nnk_lists_12 = [defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])]
    nnk_lists_12 += [defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])]
  F1_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=[M1,M1,M2], waves=waves, nnk_list=nnk_lists_12[0], diag_only=True)
  F2_diag = F_i_full_scratch(E,np.array(nnP),L, Mijk=[M2,M1,M1], waves='s', nnk_list=nnk_lists_12[1], diag_only=True)
  F_diag = F1_diag + F2_diag
  if diag_only==True:
    return F_diag
  else:
    return block_diag(*F_diag)

################################################################################
# Full ND matrix \wt{F} computed from scratch
################################################################################
def F_full_ND_scratch(E,nnP,L, M123=[1,1,1], waves='s', nnk_lists_123=None, diag_only=True):
  F_diag = []
  for i in range(3):
    Mijk = defns.get_Mijk(M123,i)
    if nnk_lists_123 == None:
      nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=Mijk)
    else:
      nnk_list = nnk_lists_123[i]
    Fi_diag = Fi_full_scratch(E,nnP,L, Mijk=Mijk, waves=waves, nnk_list=nnk_list, diag_only=True)
    F_diag += Fi_diag
  if diag_only==True:
    return F_diag
  else:
    return block_diag(*F_diag)
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
'''
The remaining code is for a faster method of (repeatedly) computing F that
splits the sum into two parts: the sum over terms near the on-shell pole, and
the sum over the remaining smooth terms. The latter is then combined with the
smooth PV integral, yielding a smooth result that permits interpolation of the
individual matrix elements as a function of the two-particle CM energy of the
nonspectator pair (for fixed masses & box size).
'''
################################################################################
# Extend scipy.interpolate.interp1d to handle F matrices
################################################################################
# def Ft_nnk_mat_interp(Pvec,kvec, Ecm_list, F_bare_list, kind='cubic'):
#   if len(Ecm_list) == 3:
#     kind = 'quadratic'
#   elif len(Ecm_list) == 2:
#     kind = 'linear'
#   elif len(Ecm_list) == 1:
#     kind = 'nearest'
#
#   def f_bare(i,j,Ecm):
#     Fij_list = np.array([F_bare[i,j] for F_bare in F_bare_list])
#     fij = interp1d(Ecm_list, Fij_list, kind=kind, bounds_error=False, fill_value='extrapolate')
#     return fij(Ecm)
#
#   def hhk(Ecm):
#     E = sqrt(Ecm**2 + sum([x**2 for x in Pvec]))
#     E2k2 = defns.E2k2(E,Pvec,kvec)
#     return defns.hh(E2k2)
#
#   out = lambda Ecm: hhk(Ecm)*np.array([[f_bare(i,j,Ecm) for j in range(6)] for i in range(6)])
#   return out

################################################################################
# List all "relevant" nna for which the on-shell energy om_k+om_a+om_b gives a
# CM energy in [Ecm_min,Ecm_max]
################################################################################
def list_nna_on(L,nnP,nnk, Mijk=[1,1,1], Ecm_min=2.8, Ecm_max=5.4):
  [Mi, Mj, Mk] = Mijk
  twopibyL = 2*pi/L
  Pvec = twopibyL*np.array(nnP); Psq = sum(Pvec**2)
  kvec = twopibyL*np.array(nnk)
  omk = sqrt(sum([x**2 for x in kvec]) + Mi**2)
  P2kvec = Pvec-kvec

  E_max = sqrt(Ecm_max**2+Psq) # should be the right E to use; it should give an upper bound on n_max
  a_max = defns.amax_Pvec_on(E_max,Pvec, Mijk=Mijk)
  n_max = int(np.floor(a_max/twopibyL))

  nna_on_list = []
  for n1 in range(-n_max, n_max+1):
    for n2 in range(-n_max, n_max+1):
      for n3 in range(-n_max, n_max+1):
        nna = np.array([n1,n2,n3])
        avec = twopibyL*nna
        bvec = P2kvec-avec
        oma = sqrt(sum([x**2 for x in avec]) + Mj**2)
        omb = sqrt(sum([x**2 for x in bvec]) + Mk**2)
        Ecm = sqrt((omk+oma+omb)**2 - Psq)
        if Ecm_min <= Ecm <= Ecm_max:
          nna_on_list.append(list(nna))
  return nna_on_list

################################################################################
# Sum only over nna close to on-shell pole
################################################################################
def sum_pole_nnk(E,nnP,L,nnk, Mijk=[1,1,1], Ecm_min=2.8, Ecm_max=5.4, waves='spd'):
  nnP = np.array(nnP); nnk = np.array(nnk)
  [Mi, Mj, Mk] = Mijk
  W = defns.get_lm_size(waves)

  twopibyL = 2*pi/L
  kvec = nnk*twopibyL
  omk = sqrt(sum(kvec**2)+Mi**2)
  E2k = E - omk
  nnP2k = nnP - nnk

  sig_i = E2k**2 - sum(nnP2k**2)*twopibyL**2
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  alpha_ij = sqrt(q2_i + Mj**2) / sqrt(sig_i)

  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])
  if hhk==0.:
    return np.zeros((W,W))

  nna_list = list_nna_on(L,nnP,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
  if len(nna_list)==0:
    return np.zeros((W,W))

  gam = E2k/sqrt(sig_i)
  x2 = q2_i/twopibyL**2

  # const = 1/(32*pi**2*L*omk*E2k)
  const = hhk/(16*pi**2*L**4*omk*E2k) # TB: assumes no flavor symmetry; divide by 2 for symmetric case
                                      # TB: convention is L for beyond_iso, L**4 for TOPT
  out = np.zeros((W,W))
  for nna in nna_list:
    nna = np.array(nna)
    if list(nnP2k)==[0,0,0]:
      rvec = nna
    else:
      rvec = nna + nnP2k * (np.dot(nna,nnP2k)/sum(nnP2k**2) * (1/gam-1) - alpha_ij/gam)
    for i1 in range(W):
      [l1,m1] = defns.lm_idx(i1)
      for i2 in range(W):
        [l2,m2] = defns.lm_idx(i2)
        out[i1,i2] += twopibyL**(l1+l2) * summand(x2,rvec, l1,m1,l2,m2)
  return out * const

################################################################################
# Compute smooth part of \wt{F}(k) (smooth part of sum - integral)
################################################################################
def sum_smooth_nnk(E,nnP,L,nnk, Mijk=[1,1,1], Ecm_min=2.8, Ecm_max=5.4):
  [Mi, Mj, Mk] = Mijk
  W = defns.get_lm_size(waves)
  twopibyL = 2*pi/L

  kvec = nnk*twopibyL
  omk = sqrt(sum(kvec**2)+Mi**2)
  E2k = E-omk
  nnP2k = nnP-nnk
  sig_i = E2k**2 - sum(nnP2k**2)*twopibyL**2
  q2_i = defns.lambda_tri(sig_i,Mj**2,Mk**2)/(4*sig_i)
  alpha_ij = sqrt(q2_i + Mj**2) / sqrt(sig_i)

  hhk = defns.hh(sig_i, Mjk=[Mj,Mk])
  if hhk==0:
    return np.zeros((W,W))

  gam = E2k/sqrt(sig_i)
  x2 = q2_i/twopibyL**2

  cutoff = get_cutoff()
  nmax_real = getnmaxreal(cutoff,hhk,gam,x2)
  nmax = int(np.ceil(nmax_real))

  nna_on_list = list_nna_on(L,nnP,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)

  out = np.zeros((W,W))
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nna = np.array([n1,n2,n3])
        if LA.norm(nna)<nmax_real and list(nna) not in nna_on_list:
          if list(nnP2k)==[0,0,0]:
            rvec = nna
          else:
            rvec = nna + nnP2k * (np.dot(nna,nnP2k)/sum(nnP2k**2) * (1/gam-1) - alpha_ij/gam)
          for i1 in range(W):
            [l1,m1] = defns.lm_idx(i1)
            for i2 in range(W):
              [l2,m2] = defns.lm_idx(i2)
              out[i1,i2] += (2*pi/L)**(l1+l2) * summand(x2,rvec, l1,m1,l2,m2)
  return out # *const


# TB: got to here

################################################################################
# Compute smooth sum - integral
################################################################################
# def F_smooth_bare_nnk(E,nnP,L,nnk, Ecm_min=2.8, Ecm_max=5.4):
#   nnP = np.array(nnP); nnk = np.array(nnk)
#   twopibyL = 2*pi/L
#   kvec = nnk*twopibyL
#   omk = sqrt(sum(kvec**2)+1)
#   E2k = E-omk
#   nnP2k = nnP-nnk
#   E2kst2 = E2k**2 - sum(nnP2k**2)*twopibyL**2
#   hhk = defns.hh(E2kst2)
#
#   # if hhk==0.:
#   #   print('hhk=0',nnP,E,E2kst2)
#   #   sys.exit()
#   #   return np.zeros((6,6))
#
#   gam = E2k/sqrt(E2kst2)
#   x2 = ( (E2k/twopibyL)**2 - sum(nnP2k**2) )/4 - 1/twopibyL**2
#
#   # const = hhk/(32*pi**2*L*omk*E2k)
#   # const = 1/(32*pi**2*L*omk*E2k)
#   const = 1/(16*pi**2*L**4*omk*E2k)    #TB: note 16 & L**4 convention of ND paper
#
#   S = sum_smooth_nnk(E,nnP,L,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#   I = int_nnk(L, gam,x2)
#   out = const*(S-I)
#   return out, hhk
#
# ################################################################################
# # Full \wt{F}(k) matrix w/ smooth part computed directly
# ################################################################################
# def F_full_nnk(E,nnP,L,nnk, Ecm_min=2.8, Ecm_max=5.4):
#   E2k2 = defns.E2k2(E,2*pi/L*np.array(nnP),2*pi/L*np.array(nnk))
#   hhk = defns.hh(E2k2)
#   pole = sum_pole_nnk(E,nnP,L,nnk, alpha=alpha, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#   smooth_bare, hhk = F_smooth_bare_nnk(E,nnP,L,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#   # print(pole)
#   # print(smooth)
#   return pole+hhk*smooth_bare
#
# ################################################################################
# # Full \wt{F} matrix computed from scratch
# ################################################################################
# def F_full_scratch(E,nnP,L, nnk_list=None, Ecm_min=2.8, Ecm_max=5.4, diag_only=True):
#   if nnk_list==None:
#     nnk_list = defns.list_nnk_nnP(E,L,nnP)
#   F_diag = []
#   for nnk in nnk_list:
#     pole = sum_pole_nnk(E,nnP,L,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#     smooth_bare, hhk = F_smooth_bare_nnk(E,nnP,L,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#     F_diag.append(pole+hhk*smooth_bare)
#   if diag_only==True:
#     return F_diag
#   else:
#     return block_diag(*F_diag)
#
# ################################################################################
# # Full \wt{F} matrix w/ interp. fn. of smooth part stored in interp_dict_nnP
# ################################################################################
# def F_full_interp(E,nnP,L,interp_dict_nnP, Ecm_min=2.8, Ecm_max=5.4, diag_only=False):
#   Ecm = sqrt(E**2 - (2*pi/L)**2 * sum([x**2 for x in nnP]))
#   F_diag_list = []
#   for nnk in defns.list_nnk_nnP(E,L,nnP):
#     pole = sum_pole_nnk(E,nnP,L,nnk, Ecm_min=Ecm_min, Ecm_max=Ecm_max)
#     # print(nnk)
#     smooth = interp_dict_nnP[tuple(nnk)](Ecm) # includes H(k)
#     # print(nnk,pole,smooth); sys.exit()
#     F_diag_list.append(pole+smooth)
#   if diag_only==True:
#     return F_diag_list
#   else:
#     return block_diag(*F_diag_list)

################################################################################
# Graveyard (old implementation of list_nna_on)
################################################################################
# def list_nnk_on(E,L,nnP):
#   twopibyL = 2*pi/L
#   Pvec = twopibyL*np.array(nnP)
#   n_max_real = defns.kmax_Pvec_on(E,Pvec)/twopibyL
#   n_max = int(np.floor(n_max_real))
#   nnk_on_list = []
#   for n1 in range(-n_max, n_max+1):
#     for n2 in range(-n_max, n_max+1):
#       for n3 in range(-n_max, n_max+1):
#         nnk = np.array([n1,n2,n3])
#         kvec = twopibyL*nnk
#         omk = sqrt(sum([x**2 for x in kvec]) + 1)
#         P2k2 = sum((Pvec-kvec)**2)
#         E2k2 = (E-omk)**2 - P2k2
#         if E2k2 >= 4.:
#           nnk_on_list.append(nnk)
#   return nnk_on_list
#
# def list_nna_on(L,nnP,nnk, Ecm_max=5.4):
#   Psq = (2*pi/L)**2*sum([x**2 for x in nnP])
#   E_max = sqrt(Ecm_max**2 + Psq)
#   omk = sqrt((2*pi/L)**2*sum([x**2 for x in nnk]) + 1)
#
#   nna_on_list = []
#   for nna in list_nnk_on(E_max,L,nnP):
#     nnb = [nnP[i]-nnk[i]-nna[i] for i in range(3)]
#     oma = sqrt((2*pi/L)**2*sum([x**2 for x in nna]) + 1)
#     omb = sqrt((2*pi/L)**2*sum([x**2 for x in nnb]) + 1)
#     Ecm = sqrt((omk+oma+omb)**2 - Psq)
#     if Ecm <= Ecm_max:
#       print(nna,nnb,Ecm)
#       nna_on_list.append(list(nna))
#   return nna_on_list
