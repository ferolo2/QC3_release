import numpy as np
pi=np.pi; LA=np.linalg
from itertools import permutations as perms
from scipy.interpolate import interp1d
from constants import *

# from numba import jit,njit

####################################################################################
# This file defines several basic functions that get called multiple times
####################################################################################
# Continue np.sqrt to handle negative arguments
#@jit(nopython=True,fastmath=True,cache=True,nogil=True)
def sqrt(x):
  if x<0:
    return np.sqrt(-x)*1j
  else:
    return np.sqrt(x)

# @jit(nopython=True,fastmath=True)
def square(x):
  return x**2

# Return unit vector in direction of vec
def vec_hat(vec):
  v = LA.norm(vec)
  if v==0:
    return vec
  else:
    return np.array(vec)/v

# om_k
# @njit(fastmath=True,cache=True)
def omega(k,m=1):
	return sqrt( k**2 + m**2 )

# # E2k*
# @njit(fastmath=True,cache=True)
# def E2k(E,Pvec,kvec):
# 	return sqrt( 1 + E**2 - 2 * E * omega(k) ) # should always be >=0

# sigma_i or (E2k*)^2
#@njit(fastmath=True,cache=True)
def sigma_i(E,Pvec,pvec_i,Mi=1):
  p = LA.norm(pvec_i)
  return (E-omega(p,Mi))**2 - LA.norm(np.array(Pvec)-np.array(pvec_i))**2  # should always be >=0

# Triangle function lambda(a,b,c)
def lambda_tri(a,b,c):
  return a**2+b**2+c**2 - 2*(a*b+a*c+b*c)

# @njit(fastmath=True,cache=True)
def qst2_i(E,Pvec,pvec_i, Mijk=[1,1,1]):
  [Mi,Mj,Mk] = Mijk
  s_i = sigma_i(E,Pvec,pvec_i, Mi=Mi)
  return lambda_tri(s_i,Mj**2,Mk**2)/(4*s_i)

################################################################################
# Spectator cutoff functions
################################################################################
# J function for spectator cutoff fn.
def jj(x):
  # (xmin, xmax) = (0.02, 0.97)
  xmin, xmax = get_xrange()
  if xmin < x < xmax:
    return np.exp(-np.exp(-1/(1-x))/x)
  elif x >= xmax:
    return 1.
  else:
    return 0.

# Spectator cutoff fn. H_i(k)=hh(sig_i)
def hh(sig_i,Mjk=[1,1]):
  epsH = get_epsH()
  [Mj,Mk] = Mjk
  #zi = (1+epsH) * sig_i / (Mj+Mk)**2  # old (from ND paper); problematic for K2
  z_i = (1+epsH) * (sig_i - abs(Mj**2-Mk**2)) / (2*min(Mj,Mk)*(Mj+Mk))   # new
  return jj(z_i)

################################################################################
# Boost pvec to rest frame of P2=(E2,P2vec)
#@njit(fastmath=True,cache=True)
def boost(p0, pvec, E2, P2vec):
  if list(P2vec) == [0,0,0]:
    return np.array(pvec)
  pvec = np.array(pvec); P2vec = np.array(P2vec)

  P2norm = sqrt(sum(P2vec**2))
  P2hat = P2vec/P2norm

  beta2 = P2norm/E2
  gam2 = 1./sqrt(1.-beta2**2)

  out = pvec + ((gam2-1)*np.dot(pvec,P2hat) - gam2*beta2*p0)*P2hat
  return out

# Convert E to Ecm or vice versa
def E_to_Ecm(E,L,nnP,rev=False):
  if rev==True:
    return sqrt(E**2 + (2*pi/L)**2*sum([x**2 for x in nnP]))
  else:
    return sqrt(E**2 - (2*pi/L)**2*sum([x**2 for x in nnP]))

##########################################################
# Return [Mi, Mj, Mk] given [M1,M2,M3] and spectator flavor index i
# Can also specify j, otherwise assumes (i,j,k) is in cyclic order
def get_Mijk(M123,i,j=None):
  if j==i:
    print("Error in get_Mijk: i and j must be different")
    print('(i,j):',(i,j))
    raise ValueError
  if len(M123) == 2:
    if i==0:
      return [M123[0], M123[0], M123[1]]
    else:
      return [M123[1], M123[0], M123[0]]
  if j==None:
    j = (i+1) % 3
  k = -(i+j) % 3
  return [M123[i],M123[j],M123[k]]

# Return number of distinct lm elements given partial waves used
def get_lm_size(waves):
  if waves=='spd':
    return 9
  elif waves=='sp':
    return 4
  elif waves=='s':
    return 1
  elif waves=='sd':
    return 6

# Convert block-matrix index to (l,m)
# @jit(nopython=True,fastmath=True,parallel=True) #FRL, it speeds up a bit. I changed the error condition to make it compatible with numba.
def lm_idx(i,waves='spd'):
  W = get_lm_size(waves)
  i = i%W
  if i==0:
    return (0,0)
  elif waves=='sd':
    return (2,i-3)
  elif 'p' in waves:
    if i<=3:
      return (1,i-2)
    elif 'd' in waves:
      return (2,i-6)

# Replace small real numbers in array with zero
def chop(arr,tol=1e-13):
  arr = np.array(arr)
  arr[abs(arr)<tol]=0
  return arr

####################################################################################
# Create nnk_list, etc.
# Permutation conventions: 000, 00a, aa0, aaa, ab0, aab, abc
####################################################################################
# Find maximum allowed norm(k) for given E,Pvec by considering the extreme case khat=Phat
#@jit(nopython=True,fastmath=True,cache=True)
def kmax_Pvec(E, Pvec, Mijk=[1,1,1]):
  [Mi, Mj, Mk] = Mijk
  Psq = sum([x**2 for x in Pvec])
  Ecm2 = E**2-Psq
  sig_i_min = abs(Mj**2-Mk**2) + 2*get_xrange()[0] * (Mj+Mk) * min(Mj,Mk) / (1+get_epsH())
  u = 0.5 + (Mi**2-sig_i_min)/(2*Ecm2)
  if u**2 - Mi**2/Ecm2 < 0:
    print('uh oh', Pvec, sig_i_min,u)
  return sqrt(Psq)*u + E*sqrt(u**2 - Mi**2/Ecm2) + 1e-14  # TB: small buffer to account for machine precision

# Find maximum norm(k) satisfying E2j*>=Mi+Mk for given E,Pvec
def amax_Pvec_on(E,Pvec, Mijk=[1,1,1]):
  [Mi, Mj, Mk] = Mijk
  P = LA.norm(Pvec)
  Ecm2 = E**2-P**2
  sig_j_min = (Mi+Mk)**2
  u = 0.5 + (Mj**2-sig_i_min)/(2*Ecm2)
  return P*u + E*sqrt(u**2 - Mj**2/Ecm2) + 1e-14    # TB: small buffer to account for machine precision

# Find energy where a certain nnk turns on for given L,nnP
# Note: epsH and xmin must match those used in kmax_Pvec
def Emin_nnP(nnk, L, nnP, Mijk=[1,1,1]):
  [Mi, Mj, Mk] = Mijk # Mk not related to nnk (notational collision)
  sig_i_min = abs(Mj**2-Mk**2) + 2*get_xrange()[0] * (Mj+Mk) * min(Mj,Mk) / (1+get_epsH())
  omk = sqrt( (2*pi/L)**2 * sum([x**2 for x in nnk]) + Mi**2 )
  Pmk2 = (2*pi/L)**2 * sum([(nnP[i]-nnk[i])**2 for i in range(3)])
  return omk + sqrt(Pmk2+sig_i_min)


################################################################################
# Create list of all permutations & all perms w/ 1 negation
def perms_list(nnk):
  a=nnk[0]; b=nnk[1]; c=nnk[2]
  p_list = list(perms((a,b,c)))
  p_list += list(perms((a,b,-c)))
  p_list += list(perms((a,-b,c)))
  p_list += list(perms((-a,b,c)))

  p_list = [ p for p in p_list ]
  return p_list


# Get Oh shell that nnk is in (w/ our formatting convention)
def get_shell(nnk):
  a,b,c = sorted([abs(x) for x in nnk])
  if a==0<b:
    return [b,c,0]
  else:
    return [a,b,c]

# Create list of all nnk in a given shell
#@jit()
def shell_nnk_list(shell):
  # 000
  if list(shell)==[0,0,0]:
    return [shell]

  # 00a
  elif shell[0]==shell[1]==0<shell[2]:
    a = shell[2]
    return [(0,0,a),(0,a,0),(a,0,0),(0,0,-a),(0,-a,0),(-a,0,0)]

  # aa0
  elif shell[0]==shell[1]>0==shell[2]:
    a = shell[0]
    return [(a,a,0),(a,0,a),(0,a,a),(a,-a,0),(a,0,-a),(0,a,-a),    (-a,-a,0),(-a,0,-a),(0,-a,-a),(-a,a,0),(-a,0,a),(0,-a,a)]

  # aaa
  elif shell[0]==shell[1]==shell[2]>0:
    a = shell[0]
    return [(a,a,a),(a,a,-a),(a,-a,a),(-a,a,a), (-a,-a,-a),(-a,-a,a),(-a,a,-a),(a,-a,-a)]

  # ab0
  elif 0==shell[2]<shell[0]<shell[1]:
    a = shell[0]; b = shell[1]
    return [
      (a,b,0),(b,a,0),(a,0,b),(b,0,a),(0,a,b),(0,b,a),
      (a,-b,0),(-b,a,0),(a,0,-b),(-b,0,a),(0,a,-b),(0,-b,a),
      (-a,-b,0),(-b,-a,0),(-a,0,-b),(-b,0,-a),(0,-a,-b),(0,-b,-a),
      (-a,b,0),(b,-a,0),(-a,0,b),(b,0,-a),(0,-a,b),(0,b,-a)
    ]

  # aab
  elif 0<shell[0]==shell[1]!=shell[2]>0:
    a = shell[0]; b = shell[2]
    return [
      (a,a,b),(a,b,a),(b,a,a),
      (a,a,-b),(a,-b,a),(-b,a,a),
      (a,-a,b),(a,b,-a),(b,a,-a),
      (-a,a,b),(-a,b,a),(b,-a,a),

      (-a,-a,-b),(-a,-b,-a),(-b,-a,-a),
      (-a,-a,b),(-a,b,-a),(b,-a,-a),
      (-a,a,-b),(-a,-b,a),(-b,-a,a),
      (a,-a,-b),(a,-b,-a),(-b,a,-a)
    ]

  # abc
  elif 0<shell[0]<shell[1]<shell[2]:
    auxshell1 = perms_list(shell)
    auxshell2 = 1*auxshell1
    for i in range(len(auxshell1)):
        auxshell2[i] = tuple([x*-1 for x in auxshell1[i]])
    return auxshell1+auxshell2

  else:
    print('Error in shell_nnk_list: Invalid shell input')


# Create list of all nnk in a given shell which are "active"
def shell_nnk_nnP_list(E,L,nnP,shell, Mijk=[1,1,1]):
  Pvec = np.array(nnP)*2*pi/L
  nnk_list=[]
  for nnk in shell_nnk_list(shell): # TB: just do this instead of truncate()
    if E > Emin_nnP(nnk, L, nnP, Mijk=Mijk): # TB: xmin needs to be same as in jj()
      #print(nnk,LA.norm(nnk))
      nnk_list.append(nnk)
  return nnk_list


# Create list of shells/orbits
#@jit(fastmath=True,cache=True) # causes very, very bizarre issues
def shell_list_nnP(E,L,nnP, Mijk=[1,1,1]):
  Pvec = 2*pi/L * np.array(nnP)
  nmaxreal = kmax_Pvec(E,Pvec, Mijk=Mijk)*L/(2*pi)
  nmax = int(np.floor(nmaxreal))
  shells = []
  for n1 in range(nmax+1):
    for n2 in range(n1,nmax+1):
      for n3 in range(n2,nmax+1):
        if square(n1)+square(n2)+square(n3) <= square(nmaxreal):
          # need to permute for aa0, ab0, aab
          if (n1==0<n2) or (n1>0 and n2==n3):
            shells.append((n2,n3,n1))
          else:
            shells.append((n1,n2,n3))
  # Sort by magnitude
  shells = sorted(shells, key=LA.norm)

  out = []
  for shell in shells:
    if len(shell_nnk_nnP_list(E,L,nnP,shell, Mijk=Mijk))>0: # TB: gross and inefficient, but it works
      out.append(shell)
  #print(out)
  return out


# New nnk_list broken into shells
def list_nnk_nnP(E,L,nnP, Mijk=[1,1,1]):
  nnk_list = []
  for shell in shell_list_nnP(E,L,nnP, Mijk=Mijk):
    nnk_list += shell_nnk_nnP_list(E,L,nnP,shell, Mijk=Mijk) # TB: use this instead of truncate()
  return nnk_list



# Get tuple(nnk_list) from precomputed dictionary at a given Ecm
def get_nnk_list(nnk_list_dict_nnP,Ecm,Ecm_max=5.2):
  if Ecm > Ecm_max:
    print('Error in get_nnk_list: Ecm outside grid range')
    print('Ecm:',Ecm)
    print('Ecm_max:',Ecm_max)
    raise ValueError
  nnk_list_tup = ()
  flag = 0
  for i,x in enumerate(nnk_list_dict_nnP):
    if Ecm < x[0]:
      if nnk_list_tup==(0,0,0):
        return [(0,0,0)]
      else:
        return list(nnk_list_tup)
    nnk_list_tup = x[1]
  # print('Warning in get_nnk_list: nnk_list may not be in dictionary')
  return list(nnk_list_tup)

####################################################################################
# Real spherical harmonics
####################################################################################
#@jit(nopython=True,fastmath=True,parallel=True)
def y1real(kvec,m): # y1 = sqrt(4pi) * |kvec| * Y1
  if m==-1:
    return sqrt(3)*kvec[1]
  elif m==0:
    return sqrt(3)*kvec[2]
  elif m==1:
    return sqrt(3)*kvec[0]
  else:
    print('Error: invalid m input in y1real')

#@jit(nopython=True,fastmath=True,parallel=True)
def y2real(kvec,m): # y2 = sqrt(4pi) * |kvec|**2 * Y2
  if m==-2:
    return sqrt(15)*kvec[0]*kvec[1]
  elif m==-1:
    return sqrt(15)*kvec[1]*kvec[2]
  elif m==0:
    return sqrt(5/4)*(2*square(kvec[2])-square(kvec[0])-square(kvec[1]))
  elif m==1:
    return sqrt(15)*kvec[0]*kvec[2]
  elif m==2:
    return sqrt(15/4)*(square(kvec[0])-square(kvec[1]))
  else:
    print('Error: invalid m input in y2real')

#@jit(nopython=True,fastmath=True,parallel=True)
# def y2(kvec,m):
#   return y2real(kvec,m)

# General spherical harmonic
#@jit(nopython=True,fastmath=True,parallel=True)
def ylm(kvec,l,m):
  if l==m==0:
    return 1
  elif l==1:
    return y1real(kvec,m)
  elif l==2:
    return y2real(kvec,m)
  else:
    print('Error: ylm can only take l=0,1,2')

################################################################################
# Extend scipy.interpolate.interp1d to handle matrices
################################################################################
def mat_interp(Ecm_list, mat_list, kind='cubic'):
  if len(Ecm_list) == 3:
    kind = 'quadratic'
  elif len(Ecm_list) == 2:
    kind = 'linear'
  elif len(Ecm_list) == 1:
    kind = 'nearest'

  def f(i,j,Ecm):
    mat_ij_list = np.array([mat[i,j] for mat in mat_list])
    fij = interp1d(Ecm_list, mat_ij_list, kind=kind, bounds_error=False, fill_value='extrapolate')
    return fij(Ecm)

  N = len(mat_list[0])
  out = lambda Ecm: np.array([[f(i,j,Ecm) for j in range(N)] for i in range(N)])
  return out

################################################################################
# Find energies in Ecm_list within max_diff of any point in Ecm_dat_list_nnP
################################################################################
def make_Ecm_list_nnP(Ecm_list, Ecm_dat_list_nnP, max_diff=0.01):
  Ecm_list_nnP = []
  idx_list_nnP = []
  for i,Ecm in enumerate(Ecm_list):
    if any([abs(Ecm-Ecm_dat)<max_diff for Ecm_dat in Ecm_dat_list_nnP]):
      Ecm_list_nnP.append(Ecm)
      idx_list_nnP.append(i)
  return Ecm_list_nnP, idx_list_nnP

####################################################################################
# Graveyard (old functions)
####################################################################################
# # Assumes nnP=(000)
# def list_nnk(E,L):
#   nnk_list = []
#   for shell in shell_list(E,L):
#     nnk_list += shell_nnk_list(shell)
#   return nnk_list
