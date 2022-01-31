import numpy as np, sys
pi=np.pi; conj=np.conjugate; LA=np.linalg
from itertools import permutations as perms
import defns; sqrt=defns.sqrt

#from projections import l0_proj, l2_proj
from scipy.linalg import block_diag


####################################################################################
# Group theory definitions
####################################################################################

############################################################
# Basic transformation functions
############################################################
# Apply R_p to vec for permutation vector p; e.g. R_{31-2}[0,1,3] = [3,0,-1]
def cubic_transf(vec,p):
  return [np.sign(i)*vec[abs(i)-1] for i in p]

# Find [i,j,k] s.t. R_{ijk}p1 = p2
def Rvec(p2,p1):
  abs_p1 = [abs(x) for x in p1]
  abs_p2 = [abs(x) for x in p2]
  i0 = abs_p1.index(abs_p2[0])
  i1 = abs_p1.index(abs_p2[1])
  if i1==i0:
    i1 = abs_p1.index(abs_p2[1],i0+1,3)
  i2 = [i for i in range(3) if i not in (i0,i1)][0]
  if abs_p1[i2] != abs_p2[2]:
    print('Error in Rvec')

  ivec = [i0,i1,i2]
  R=[]
  for j in range(3):
    sgn = 1
    if np.sign(p2[j]) != np.sign(p1[ivec[j]]):
      sgn = -1
    R.append(sgn*(ivec[j]+1))
  return R

# Compute p=[i,j,k] = ...p2*p1 s.t. R_p = ...R_{p2}R_{p1}
def R_prod(*argv):
  p = [1,2,3]
  for j in range(len(argv)-1,-1,-1):
    p2 = argv[j]
    p = [np.sign(i)*p[abs(i)-1] for i in p2]
  return p


###################################################################################
# p-wave Wigner D-matrices D(R) in real Ylm basis (ouputs 3x3 p-wave matrix)
###################################################################################
def Dmat11(R):
  R = list(R)
  if tuple(abs(i) for i in R) not in list(perms([1,2,3])):
    print('Error in Dmat11: invalid input')

  # Trivial transformation:
  if R==[1,2,3] or R==[-1,-2,-3]:
    return np.sign(R[0]) * np.eye(3)

  # Single permutation:
  elif R==[2,1,3] or R==[-2,-1,-3]:
    U = np.eye(3)
    U[0][0]=0; U[2][2]=0;
    U[0][2]=1; U[2][0]=1;
    return np.sign(R[0]) * U
  elif R==[1,3,2] or R==[-1,-3,-2]:
    U = np.eye(3)
    U[0][0]=0; U[1][1]=0;
    U[0][1]=1; U[1][0]=1;
    return np.sign(R[0]) * U
  elif R==[3,2,1] or R==[-3,-2,-1]:
    U=np.eye(3)
    U[1][1]=0; U[2][2]=0;
    U[1][2]=1; U[2][1]=1;
    return np.sign(R[0]) * U

  # Cyclic permutation:
  elif R==[2,3,1] or R==[-2,-3,-1]:
    return np.sign(R[0]) * defns.chop( Dmat11([1,3,2]) @ Dmat11([2,1,3]) )
  elif R==[3,1,2] or R==[-3,-1,-2]:
    return np.sign(R[0]) * defns.chop( Dmat11([3,2,1]) @ Dmat11([2,1,3]) )

  # Single negation:
  elif R==[1,2,-3] or R==[-1,-2,3]:
    return np.sign(R[0]) * defns.chop( np.diag([1,-1,1]) )
  elif R[0]*R[1]>0 and R[0]*R[2]<0:
    return defns.chop( Dmat11([1,2,-3]) @ Dmat11([R[0],R[1],-R[2]]) )
  elif R[0]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat11([1,3,2]) @ Dmat11([R[0],R[2],R[1]]) )
  elif R[1]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat11([3,2,1]) @ Dmat11([R[2],R[1],R[0]]) )

  else:
    print('Error in Dmat11: This should never trigger')


###################################################################################
# d-wave Wigner D-matrices D(R) in real Ylm basis (ouputs 5x5 d-wave matrix)
###################################################################################
def Dmat22(R):
  R = list(R)
  if tuple(abs(i) for i in R) not in list(perms([1,2,3])):
    print('Error in Dmat22: invalid input')

  # Trivial transformation:
  if R==[1,2,3] or R==[-1,-2,-3]:
    return np.identity(5)

  # Single permutation:
  elif R==[2,1,3] or R==[-2,-1,-3]:
    U = np.identity(6)
    U[2][2]=0; U[2][4]=1
    U[4][2]=1; U[4][4]=0
    U[5][5]=-1
    return U[1:,1:]

  elif R==[1,3,2] or R==[-1,-3,-2]:
    U = np.zeros((6,6))
    U[0][0]=1
    U[1][4]=1; U[2][2]=1; U[4][1]=1
    U[3][3] = -1/2;       U[3][5] = -sqrt(3)/2
    U[5][3] = -sqrt(3)/2; U[5][5] = 1/2
    return U[1:,1:]

  elif R==[3,2,1] or R==[-3,-2,-1]:
    U = np.zeros((6,6))
    U[0][0]=1
    U[1][2]=1; U[2][1]=1; U[4][4]=1
    U[3][3] = -1/2;       U[3][5] = sqrt(3)/2
    U[5][3] = sqrt(3)/2; U[5][5] = 1/2
    return U[1:,1:]

  # Cyclic permution:
  elif R==[2,3,1] or R==[-2,-3,-1]:
    return defns.chop( Dmat22([1,3,2]) @ Dmat22([2,1,3]) )

  elif R==[3,1,2] or R==[-3,-1,-2]:
    return defns.chop( Dmat22([3,2,1]) @ Dmat22([2,1,3]) )

  # Single negation
  elif R==[1,2,-3] or R==[-1,-2,3]:
    return defns.chop( np.diag([1,-1,1,-1,1]) )

  elif R[0]*R[1]>0 and R[0]*R[2]<0:
    return defns.chop( Dmat22([1,2,-3]) @ Dmat22([R[0],R[1],-R[2]]) )

  elif R[0]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat22([1,3,2]) @ Dmat22([R[0],R[2],R[1]]) )

  elif R[1]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat22([3,2,1]) @ Dmat22([R[2],R[1],R[0]]) )

  else:
    print('Error in Dmat22: This should never trigger')
###################################################################################
# Wigner D-matrices D(R) in real Ylm basis (ouputs 4x4 s-wave \oplus p-wave matrix)
###################################################################################
def Dmat(R):
  Dmat = np.zeros((4,4))
  Dmat[0,0] = 1
  Dmat[1:,1:] = Dmat11(R)
  return Dmat

############################################################
# Cubic groups, little groups
############################################################
# Create list of 24 pure rotations (i.e. point group O w/o inversions)
def rotations_list():
  out = [[1,2,3],[2,3,1],[3,1,2],[1,3,-2],[2,-1,3],[3,2,-1],
          [1,-2,-3],[2,-3,-1],[3,-1,-2],[1,-3,2],[2,1,-3],[3,-2,1],
          [-1,2,-3],[-2,3,-1],[-3,1,-2],[-1,3,2],[-2,-1,-3],[-3,2,1],
          [-1,-2,3],[-2,-3,1],[-3,-1,2],[-1,-3,-2],[-2,1,3],[-3,-2,-1]]
  return out

# Create list of all 48 permutations w/ any # of negations (i.e., full cubic group w/ inversions Oh)
def Oh_list():
  Oh_list = list(perms([1,2,3]))
  Oh_list += list(perms([1,2,-3]))
  Oh_list += list(perms([1,-2,3]))
  Oh_list += list(perms([-1,2,3]))
  Oh_list += list(perms([1,-2,-3]))
  Oh_list += list(perms([-1,2,-3]))
  Oh_list += list(perms([-1,-2,3]))
  Oh_list += list(perms([-1,-2,-3]))

  Oh_list = [ list(R) for R in Oh_list ]
  return Oh_list

# Little group for given shell type
def little_group(shell):
  shell = list(shell)
  # 000
  if shell==[0,0,0]:
    return Oh_list()
  # 00a
  elif shell[0]==shell[1]==0:
    return [[1,2,3],[-1,2,3],[1,-2,3],[-1,-2,3],[2,1,3],[-2,1,3],[2,-1,3],[-2,-1,3]]
  # aa0
  elif shell[0]==shell[1]!=shell[2]==0:
    return [[1,2,3],[1,2,-3],[2,1,3],[2,1,-3]]
  # aaa
  elif shell[0]==shell[1]==shell[2]:
    return [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
  # ab0
  elif 0!=shell[0]!=shell[1]!=0==shell[2]:
    return [[1,2,3],[1,2,-3]]
  # aab
  elif shell[0]==shell[1]!=shell[2]:
    return [[1,2,3],[2,1,3]]
  # abc
  elif 0!=shell[0]!=shell[1]!=shell[2]!=shell[0]!=0 and shell[1]!=0:
    return [[1,2,3]]
  else:
    print('Error: invalid shell input')
    sys.exit()


###########################################################
# Irreps, conjugacy classes, characters
###########################################################
# List of irreps of each little group LG(Pvec)
def irrep_list(Pvec):
  Pvec = list(Pvec)

  # 000 (Oh)
  if Pvec==[0,0,0]:
    return ['A1g','A2g','Eg','T1g','T2g','A1u','A2u','Eu','T1u','T2u']

  # 00a (C4v)
  elif Pvec[0]==Pvec[1]==0:
    return ['A1','A2','B1','B2','E']

  # aa0 (C2v)
  elif Pvec[0]==Pvec[1]!=Pvec[2]==0:
    return ['A1','A2','B1','B2']

  # aaa (C3v)
  elif Pvec[0]==Pvec[1]==Pvec[2]:
    return ['A1','A2','E']

  # ab0, aab (C2)
  elif 0!=Pvec[0]!=Pvec[1]!=0==Pvec[2] or Pvec[0]==Pvec[1]!=Pvec[2]:
    return ['A1','A2']

  # abc (C1 = trivial)
  # elif 0!=Pvec[0]!=Pvec[1]!=Pvec[2]!=Pvec[0]!=0 and Pvec[1]!=0:
  #   return []

  else:
    print('Error: invalid Pvec input')

# Dimension of irrep
def irrep_dim(I):
  if I in ['A1g','A1','A2g','A2','A1u','A2u','B1','B2']:
    return 1
  elif I in ['Eg','E','Eu','E2']:
    return 2
  elif I in ['T1g','T1','T2g','T2','T1u','T2u']:
    return 3
  else:
    print('Error: invalid irrep in irrep_dim -- "{}"'.format(I))

# Format irrep strings to be Latex-friendly (not including $ signs)
def irrep_tex(irrep):
  if len(irrep) > 1:
    # if irrep[-1] == '+':
    #   irrep = irrep[:-1]+'g'
    # elif irrep[-1] == '-':
    #   irrep = irrep[:-1]+'u'
    irrep = irrep[0]+'_{'+irrep[1:]+'}'
  return irrep

# Compute conjugacy class of R_p, where p=[i,j,k]
def conj_class(p):
  p = list(p)
  p_abs = [abs(x) for x in p]

  N_negs = sum([1 for x in range(3) if p[x]<0])
  N_correct = sum([1 for x in range(3) if p_abs[x]==x+1])

  if N_correct == 3:
    if N_negs == 0:
      return 'E'
    elif N_negs == 2:
      return 'C4^2'

    elif N_negs == 3:
      return 'i'
    elif N_negs == 1:
      return 'sigma_h'

  elif N_correct == 0:
    if (N_negs % 2) == 0:
      return 'C3'
    else:
      return 'S6'

  elif N_correct == 1:
    i_correct = [i for i in range(3) if p_abs[i]==i+1][0]
    if (N_negs % 2) == 1:
      if p[i_correct] < 0:
        return 'C2'
      else:
        return 'C4'
    else:
      if p[i_correct] > 0:
        return 'sigma_d'
      else:
        return 'S4'
  else:
    print('Error in conj_class: should never reach here')


# Compute character for R_p in irrep I of LG(Pvec), where p=[i,j,k]
def chi(p,I,Pvec):
  Pvec = list(Pvec)
  cc = conj_class(p)

  # 000 (Oh)
  if Pvec==[0,0,0]:
    if I in ['A1g','A1']:
      return 1

    elif I in ['A2g','A2']:
      if cc in ['C2','C4','sigma_d','S4']:
        return -1
      else:
        return 1

    elif I in ['Eg','E']:
      if cc in ['E','C4^2','i','sigma_h']:
        return 2
      elif cc in ['C3','S6']:
        return -1
      else:
        return 0

    elif I in ['T1g','T1']:
      if cc in ['E','i']:
        return 3
      elif cc in ['C3','S6']:
        return 0
      elif cc in ['C4','S4']:
        return 1
      else:
        return -1

    elif I in ['T2g','T2']:
      if cc in ['E','i']:
        return 3
      elif cc in ['C3','S6']:
        return 0
      elif cc in ['C2','sigma_d']:
        return 1
      else:
        return -1

    elif I=='A1u':
      if cc in ['E','C3','C4^2','C4','C2']:
        return 1
      else:
        return -1

    elif I=='A2u':
      if cc in ['E','C3','C4^2','S4','sigma_d']:
        return 1
      else:
        return -1

    elif I=='Eu':
      if cc in ['E','C4^2']:
        return 2
      elif cc in ['i','sigma_h']:
        return -2
      elif cc=='S6':
        return 1
      elif cc=='C3':
        return -1
      else:
        return 0

    elif I=='T1u':
      if cc=='E':
        return 3
      elif cc=='i':
        return -3
      elif cc in ['C4','sigma_h','sigma_d']:
        return 1
      elif cc in ['C2','C4^2','S4']:
        return -1
      else:
        return 0

    elif I=='T2u':
      if cc=='E':
        return 3
      elif cc=='i':
        return -3
      elif cc in ['C2','S4','sigma_h']:
        return 1
      elif cc in ['C4','C4^2','sigma_d']:
        return -1
      else:
        return 0


  # 00a (C4v)
  elif Pvec[0]==Pvec[1]==0:
    if I in ['A1']:
      return 1

    elif I in ['A2']:
      if cc in ['sigma_h','sigma_d']:
        return -1
      elif cc in ['E','C4','C4^2']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['B1']:
      if cc in ['C4','sigma_d']:
        return -1
      elif cc in ['E','C4^2','sigma_h']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['B2']:
      if cc in ['C4','sigma_h']:
        return -1
      elif cc in ['E','C4^2','sigma_d']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['E','E2']:
      if cc in ['E']:
        return 2
      elif cc in ['C4^2']:
        return -2
      elif cc in ['C4','sigma_h','sigma_d']:
        return 0
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))


  # aa0 (C2v)
  elif Pvec[0]==Pvec[1]!=Pvec[2]==0:
    if I in ['A1']:
      return 1

    elif I in ['A2']:
      if cc in ['sigma_h','sigma_d']:
        return -1
      elif cc in ['E','C2']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['B1']:
      if cc in ['C2','sigma_h']:  # TB: I had B1 and B2 reversed before
        return -1
      elif cc in ['E','sigma_d']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['B2']:
      if cc in ['C2','sigma_d']:  # TB: I had B1 and B2 reversed before
        return -1
      elif cc in ['E','sigma_h']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))


  # aaa (C3v)
  elif Pvec[0]==Pvec[1]==Pvec[2]:
    if I in ['A1']:
      return 1

    elif I in ['A2']:
      if cc in ['sigma_d']:
        return -1
      elif cc in ['E','C3']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

    elif I in ['E','E2']:
      if cc in ['E']:
        return 2
      elif cc in ['C3']:
        return -1
      elif cc in ['sigma_d']:
        return 0
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

  # ab0, aab (C2)
  elif 0!=Pvec[0]!=Pvec[1]!=0==Pvec[2] or Pvec[0]==Pvec[1]!=Pvec[2]:
    if I in ['A1','A']:
      return 1

    elif I in ['A2','B']:
      if cc in ['sigma_h','sigma_d']: # sigma_h for ab0, sigma_d for aab
        return -1
      elif cc in ['E']:
        return 1
      else:
        print('Error in chi: {} not in LG({})'.format(cc,Pvec))

  # abc (C1 = trivial)
  # elif 0!=Pvec[0]!=Pvec[1]!=Pvec[2]!=Pvec[0]!=0 and Pvec[1]!=0:
  #   return []

  else:
    print('Error: invalid Pvec input')

# Give irrep with opposite parity to I
def parity_irrep(Pvec,I):
  Pvec = list(Pvec)

  # 000 (Oh)
  if Pvec==[0,0,0]:
    if I[-1]=='g':
      return I[:-1]+'u'
    elif I[-1]=='u':
      return I[:-1]+'g'
    else:
      return I

  # 00a (C4v)
  elif Pvec[0]==Pvec[1]==0:
    if I[0]=='E':
      return I
    elif I[-1]=='1':
      return I[:-1]+'2'
    elif I[-1]=='2':
      return I[:-1]+'1'

  # aa0 (C2v)
  elif Pvec[0]==Pvec[1]!=Pvec[2]==0:
    if I[-1]=='1':
      return I[:-1]+'2'
    elif I[-1]=='2':
      return I[:-1]+'1'

  # aaa (C3v)
  elif Pvec[0]==Pvec[1]==Pvec[2]:
    if I[0]=='E':
      return I
    elif I[-1]=='1':
      return I[:-1]+'2'
    elif I[-1]=='2':
      return I[:-1]+'1'

  # ab0, aab (C2)
  elif 0!=Pvec[0]!=Pvec[1]!=0==Pvec[2] or Pvec[0]==Pvec[1]!=Pvec[2]:
    if I[:2]=='A1':
      return 'A2'+I[2:]
    elif I[:2]=='A2':
      return 'A1'+I[2:]

  # abc (C1 = trivial)
  # elif 0!=Pvec[0]!=Pvec[1]!=Pvec[2]!=Pvec[0]!=0 and Pvec[1]!=0:
  #   return []

  else:
    print('Error: invalid Pvec input')

################################################################################
# Functions for finding free energies & decomposing into irreps
################################################################################
# Sort list according to system's symmetry
def sym_sort(my_list, sym, key=None):
  # if type(my_list[0]) == list:
  #   my_key = lambda vec: sum([x**2 for x in vec])
  # else:
  #   my_key = None
  if sym=='ID':
    my_list.sort(key=key)
  return tuple(my_list)

# Find all 2pt. free energies with Ecm below Ecm_max for given L
# Output is dictionary of form dict[(level,shell)] = [Ecm,degen,configs]
def free_levels_dict_2pt(M12,L,nnP,Ecm_max=4,sym='ID'):
  [M1,M2] = M12
  M0 = min(M12)
  if not((sym=='ID' and M1==M2) or sym=='ND'):
    raise ValueError('Error: masses {} inconsistent with {} symmetry'.format(M12,sym))

  norm2 = lambda vec: sum([x**2 for x in vec])
  Emax = sqrt(Ecm_max**2 + (2*pi/L)**2*norm2(nnP))
  nmax = int(L/(2*pi) * sqrt(Emax*(Emax-2*M0))) # from assuming assuming 1 pt. at rest

  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        if n1**2+n2**2+n3**2 <= (L/(2*pi))**2*Emax*(Emax-2*M0):
          nvec_list.append([n1,n2,n3])

  level_dict = {}
  for i1 in range(len(nvec_list)):
    #for i2 in range(len(nvec_list)):
      nvec1 = nvec_list[i1]
      nvec2 = [nnP[i]-nvec1[i] for i in range(3)]

      config = sym_sort([nvec1,nvec2], sym, key=lambda vec: (norm2(vec), vec))  # sort by norm, then subsort if norms are equal (using whatever python's convention is)

      n1_2 = sum([x**2 for x in config[0]])
      n2_2 = sum([x**2 for x in config[1]])

      E1 = sqrt((2*pi/L)**2*n1_2 + M1**2)
      E2 = sqrt((2*pi/L)**2*n2_2 + M2**2)

      E = E1+E2
      Ecm = sqrt(E**2-(2*pi/L)**2*sum([x**2 for x in nnP]))

      if Ecm <= Ecm_max:
        # level = sym_sort([n1_2,n2_2], sym)
        # shell = sym_sort([tuple(sorted([abs(x) for x in nvec])) for nvec in config], sym, key=norm2)
        level = (n1_2,n2_2)
        shell = tuple([tuple(sorted([abs(x) for x in nvec])) for nvec in config])
        key = (level,shell)
        if key in level_dict:
          if config not in level_dict[key][2]:
            level_dict[key][1] += 1
            level_dict[key][2].append(config)
        else:
          level_dict[key] = [Ecm, 1, [config]]

  tmp = sorted(level_dict.items(), key=lambda x: x[1][0]) # sort by Ecm
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


# Find all 3-pt. free energies with Ecm below Ecm_max for given L
# Output is dictionary of form dict[(level,shell)] = [Ecm,degen,configs]
def free_levels_dict_3pt(M123,L,nnP,Ecm_max=5,sym='ID'):
  [M1,M2,M3] = M123 #sorted(M123)
  M0 = min(M123)
  if not((sym=='ID' and M1==M2==M3) or (sym=='2+1' and M1==M2!=M3) or sym=='ND'):
    raise ValueError('Error: masses {} inconsistent with {} symmetry'.format(M123,sym))

  norm2 = lambda vec: sum([x**2 for x in vec])
  Emax = sqrt(Ecm_max**2 + (2*pi/L)**2*norm2(nnP))
  nmax = int(L/(2*pi) * sqrt((Emax-M0)*(Emax-3*M0))) # from assuming assuming 2 pts. at rest

  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        if n1**2+n2**2+n3**2 <= (L/(2*pi))**2*(Emax-M0)*(Emax-3*M0):
          nvec_list.append([n1,n2,n3])

  level_dict = {}
  for i1 in range(len(nvec_list)):
    i2_min = i1 if sym in ['ID','2+1'] else 0
    for i2 in range(i2_min, len(nvec_list)):
      nvec1 = nvec_list[i1]
      nvec2 = nvec_list[i2]
      nvec12 = [nnP[i]-nvec1[i]-nvec2[i] for i in range(3)]

      config = sym_sort([nvec1,nvec2,nvec12], sym, key=lambda vec: (norm2(vec), vec))  # sort by norm, then subsort if norms are equal (using whatever python's convention is)

      n1_2 = sum([x**2 for x in config[0]])
      n2_2 = sum([x**2 for x in config[1]])
      n12_2 = sum([x**2 for x in config[2]])

      E1 = sqrt((2*pi/L)**2*n1_2 + M1**2)
      E2 = sqrt((2*pi/L)**2*n2_2 + M2**2)
      E12 = sqrt((2*pi/L)**2*n12_2 + M3**2)

      E = E1+E2+E12
      Ecm = sqrt(E**2-(2*pi/L)**2*sum([x**2 for x in nnP]))

      if Ecm <= Ecm_max:
        #print(nnP,config, [n1_2,n2_2,n12_2])
        # level = sym_sort([n1_2,n2_2,n12_2], sym)
        # shell = sym_sort([tuple(sorted([abs(x) for x in nvec])) for nvec in config], sym, key=norm_sort)
        level = (n1_2,n2_2,n12_2)
        shell = tuple([tuple(sorted([abs(x) for x in nvec])) for nvec in config])
        key = (level,shell)
        if key in level_dict:
          if config not in level_dict[key][2]:
            level_dict[key][1] += 1
            level_dict[key][2].append(config)
        else:
          level_dict[key] = [Ecm, 1, [config]]

  tmp = sorted(level_dict.items(), key=lambda x: x[1][0]) # sort by Ecm
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


'''Computes character table of the (generally reducible) representation of a
particular free energy level (& shell if there's an accidental degeneracy).
Outputs a dictionary of the form dict[cc] = (n,x,p), where n is the size of the
conjugacy class cc, x is the character, and p describes a transformation in
LG(nnP) under which one or more configs in config_list are invariant'''
def level_characters(nnP, config_list, sym='ID', parity=-1):
  LG = little_group(nnP)
  cc_table = {}
  N_pt = len(config_list[0])
  norm2 = lambda vec: sum([x**2 for x in vec])
  for p in LG:
    cc = conj_class(p)
    if cc not in cc_table:
      cc_table[cc] = [1,0,p]
      for config in config_list:
        new_config = sym_sort([cubic_transf(vec,p) for vec in config], sym, key=lambda vec: (norm2(vec), vec)) # sort by norm, then subsort if norms are equal (using whatever python's convention is)
        if new_config == config:
          cc_table[cc][1] += 1
      if cc[0] in 'siS':
        cc_table[cc][1] = parity**N_pt * cc_table[cc][1]  # Note: assumes odd intrinsic parity by default
    else:
      cc_table[cc][0] += 1
  #print(config_list, cc_table)
  #sys.exit()
  return cc_table

# Computes irrep decomposition given characters of an arbitrary representation
# If tex=True, gives output in Latex syntax
def irrep_decomp(nnP,cc_table, tex=False):
  N = len(little_group(nnP))
  out = ''
  irreps = []
  N_configs = cc_table['E'][1]
  total_weights = 0
  for I in irrep_list(nnP):
    prod = 0
    for cc in cc_table:
      (n,x,p) = cc_table[cc]
      prod += n*x * chi(p,I,nnP)
    weight = prod/N
    if abs(weight-round(weight)) > 1e-14:
      ValueError('Error in GT.irrep_decomp: non-integer irrep weight for nnP={}, I={} (weight={}) \n cc_table={}'.format(nnP,I,weight,cc_table))
    else:
      weight = round(weight)
    if weight != 0:
      #print(weight)
      total_weights += weight * irrep_dim(I)
      if tex == True:
        if weight==1:
          out += '{} \\oplus '.format(irrep_tex(I))
        else:
          out += '{}{} \\oplus '.format(int(weight),irrep_tex(I))
      else:
        #out += '{}*{} + '.format(weight,I)
        irreps.append((I,weight))
  if N_configs != total_weights:
    ValueError('Error in GT.irrep_decomp: number of configs inconsistent with irrep decomp -- N_configs={}, decomp={} (total_weights={})'.format(N_configs,irreps,total_weights))
  if tex == True:
    return '${}$'.format(out[:-8])
  else:
    return irreps #out[:-3]


# Compute all free 2-pt. CM energy levels below Ecm_max, including degeneracies & irrep decomps
def free_levels_decomp_2pt(M12,L,nnP,Ecm_max=4,sym='ID',parity=-1,tex=False):
  level_dict = free_levels_dict_2pt(M12,L,nnP,Ecm_max=Ecm_max,sym=sym)
  Ecm_decomp_list = []
  for key in level_dict:
    (level, shell) = key
    (Ecm, degen, configs) = level_dict[key]
    # print(level,Ecm,degen)
    cc_table = level_characters(nnP,configs, sym=sym,parity=parity)
    decomp = irrep_decomp(nnP,cc_table, tex=tex)
    Ecm_decomp_list.append((Ecm,degen,decomp))
  return Ecm_decomp_list


# Compute all free 3-pt. CM energy levels below Ecm_max, including degeneracies & irrep decomps
def free_levels_decomp_3pt(M123,L,nnP,Ecm_max=5,sym='ID',parity=-1,tex=False):
  level_dict = free_levels_dict_3pt(M123,L,nnP,Ecm_max=Ecm_max,sym=sym)
  Ecm_decomp_list = []
  for key in level_dict:
    (level, shell) = key
    (Ecm, degen, configs) = level_dict[key]
    # print(level,Ecm,degen)
    cc_table = level_characters(nnP,configs, sym=sym,parity=parity)
    decomp = irrep_decomp(nnP,cc_table, tex=tex)
    Ecm_decomp_list.append((Ecm,degen,decomp))
  return Ecm_decomp_list
