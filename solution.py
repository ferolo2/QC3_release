######################################A##########################################
# Import relevant python modules
################################################################################
# Standard python modules
import numpy as np, sys, os
np.set_printoptions(precision=6)
pi = np.pi; sqrt=np.sqrt; LA=np.linalg

# Set up paths for loading QC modules
from pathlib import Path
cwd = Path(os.getcwd())
sys.path.insert(0,str(cwd/'base_code'))
sys.path.insert(0,str(cwd/'base_code/F3'))
sys.path.insert(0,str(cwd/'base_code/Kdf3'))

# QC modules
import defns, group_theory_defns as GT, projections as proj
from F3 import G_mov, F_fast, K2i_mat, qcot_fits, F3_mat
from Kdf3 import K3main
from scipy.optimize import fsolve
################################################################################

################################################################################
# Define 2+1 system parameters --- *ALWAYS SET M1=1*
################################################################################
M1,M2 = [1.,0.5];  # The 3-pt. system masses are [M1,M1,M2]
M12 = [1.0,M2/M1]  # We always rescale by M1
parity = -1        # Particle parity (-1 for pseudoscalars)
L = 5              # Box size (in units of 1/M1)
nnP = [0,0,0]      # 3-pt. FV spatial momentum (integer-valued)                                                                                                                                    
################################################################################
# Define K2^{-1} parameters
################################################################################
waves = 'sp'  # Partial waves used for dimers with flavor-1 spectators                                                                                    
              # (flavor-2 spectators only use s-wave)                                                                                                                                     
a_1s = 0.15    # s-wave scattering length for spectator-flavor-1 channel                                                                                                 
r_1s = 0.0    # s-wave effective range for spectator-flavor-1 channel                                                                                                 
a_1p = 0.2    # p-wave scattering length for spectator-flavor-1 channel                                                                                                                   
a_2s = 0.1    # s-wave scattering length for spectator-flavor-2 channel
################################################################################
# Define K2^{-1} phase shift functions
################################################################################
f_qcot_1s = lambda q2: qcot_fits.qcot_fit_s(q2,[a_1s,r_1s],ERE=True)
f_qcot_1p = lambda q2: qcot_fits.qcot_fit_p(q2,a_1p)
f_qcot_1sp = [f_qcot_1s, f_qcot_1p]
f_qcot_2s = [lambda q2: qcot_fits.qcot_fit_s(q2,[a_2s],ERE=True)]
################################################################################
# Define Kdf3 parameters
################################################################################
K3iso = [200, 400]      # Isotropic term is K3iso[0] + K3iso[1]*\Delta
K3B_par = 400          # Parameter for K3B term
K3E_par = 300          # Parameter for K3E term
################################################################################
# Define function that returns the smallest eigenvalue of QC in an irrep
################################################################################
def QC3(E, L, nnP, f_qcot_1sp, f_qcot_2s, M12, waves, K3iso, K3B_par, K3E_par, parity, irrep):
  F3 = F3_mat.F3mat_2plus1(E,L,nnP, f_qcot_1sp,f_qcot_2s, M12=M12,waves=waves)
  K3 = K3main.K3mat_2plus1(E,L,nnP, K3iso,K3B_par,K3E_par, M12=M12,waves=waves)
  F3i = LA.inv(F3)
  QC3_mat = F3i + K3
  QC3_mat_I = proj.irrep_proj_2plus1(QC3_mat,E,L,nnP,irrep, M12=M12, waves=waves, parity=parity)
  eigvals = sorted(defns.chop(LA.eigvals(QC3_mat_I).real,tol=1e-9), key=abs)
  return eigvals[0]
################################################################################
# Define energy levels to find
################################################################################
Efree_list = [2*M1+M2] + [2*sqrt(M1**2+(2*pi/L)**2)+M2]*2 #Non-interacting energies
irrep_list = ['A1-']*2 + ['E-']
print('Non-interacting energies:')
print(np.array(Efree_list))
print('Irreps:')
print(irrep_list)
################################################################################
# Find solutions    
################################################################################
func = lambda E: QC3(E[0], L, nnP, f_qcot_1sp, f_qcot_2s, M12,  #Create wrapper
                     waves, K3iso, K3B_par, K3E_par, parity, irrep)
for i in range(len(Efree_list)):
  Efree = Efree_list[i]
  irrep = irrep_list[i]
  Etest = Efree+0.06 #Try an energy slightly above Efree
  Esol = fsolve(func, Etest)
  print('irrep:',irrep, ' solution: ', Esol)
