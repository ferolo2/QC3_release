######################################A##########################################
# Import relevant python modules
################################################################################
# Standard python modules
import numpy as np, sys, os
np.set_printoptions(precision=6)
pi = np.pi; sqrt=np.sqrt; LA=np.linalg
from scipy.optimize import fsolve

# Set up paths for loading QC modules
from pathlib import Path
cwd = Path(os.getcwd())
sys.path.insert(0,str(cwd/'base_code'))
sys.path.insert(0,str(cwd/'base_code/F3'))
sys.path.insert(0,str(cwd/'base_code/Kdf3'))

# QC modules
import defns, group_theory_defns as GT, projections as proj
from F3 import qcot_fits, F3_mat #, F_mat, G_mat, K2i_mat,
from Kdf3 import K3main
################################################################################

################################################################################
# Define 2+1 system parameters --- *MUST ALWAYS RESCALE SO THAT M1=1*
################################################################################
M1,M2 = [100.,50.]  # The 3-pt. system masses are [M1,M1,M2], e.g. in MeV
M1,M2 = [1.,M2/M1]  # We always rescale by M1 to make everything dimensionless
M12 = [M1,M2]
parity = -1         # Particle parity (-1 for pseudoscalars)
L = 5               # Box size (in units of 1/M1)
nnP = [0,0,0]       # 3-pt. FV spatial momentum (integer-valued)
################################################################################
# Define K2^{-1} parameters
################################################################################
waves = 'sp'  # Partial waves used for dimers with flavor-1 spectators
              # (flavor-2 spectators only use s-wave)
a_1s = 0.15   # s-wave scattering length for spectator-flavor-1 channel
r_1s = 0.     # s-wave effective range for spectator-flavor-1 channel
a_1p = 0.2    # p-wave scattering length for spectator-flavor-1 channel
a_2s = 0.1    # s-wave scattering length for spectator-flavor-2 channel
r_2s = 0.     # s-wave effective range for spectator-flavor-2 channel
################################################################################
# Define K2^{-1} phase shift functions
################################################################################
f_qcot_1s = lambda q2: qcot_fits.qcot_fit_s(q2,[a_1s,r_1s],ERE=True)
f_qcot_1p = lambda q2: qcot_fits.qcot_fit_p(q2,a_1p)
f_qcot_1sp = [f_qcot_1s, f_qcot_1p]
f_qcot_2s = [lambda q2: qcot_fits.qcot_fit_s(q2,[a_2s,r_2s],ERE=True)]
################################################################################
# Define Kdf3 parameters
################################################################################
K3iso = [200, 400]      # Isotropic term is K3iso[0] + K3iso[1]*\Delta
K3B_par = 400          # Parameter for K3B term
K3E_par = 300          # Parameter for K3E term
################################################################################
# Define function that returns the smallest eigenvalue of QC in an irrep
################################################################################
def QC3(Ecm, L, nnP, f_qcot_1sp, f_qcot_2s, M12, waves,
        K3iso, K3B_par, K3E_par, parity, irrep):
  E = defns.E_to_Ecm(Ecm,L,nnP, rev=True)
  F3 = F3_mat.F3mat_2plus1(E,L,nnP, f_qcot_1sp,
                           f_qcot_2s, M12=M12, waves=waves)
  K3 = K3main.K3mat_2plus1(E,L,nnP, K3iso,
                           K3B_par,K3E_par, M12=M12, waves=waves)
  F3i = LA.inv(F3)
  QC3_mat = F3i + K3
  QC3_mat_I = proj.irrep_proj_2plus1(QC3_mat,E,L,nnP,irrep,
                                     M12=M12,waves=waves,parity=parity)
  eigvals = sorted(defns.chop(LA.eigvals(QC3_mat_I).real,tol=1e-9), key=abs)
  return eigvals[0]
################################################################################
# Determine energy levels & irreps to include
################################################################################
#Ecm_max = (2*M1+M2) + 2*min(M1,M2) # This is the formal limit of validity of the QC3
Ecm_max = 3.8                       # We often explore higher energies though

# Find all free CM energy levels below Ecm_max, including degeneracies & irrep decomps
Ecm_free_decomp_list = GT.free_levels_decomp_3pt([M1,M1,M2],L,nnP,Ecm_max=Ecm_max,sym='2+1',parity=parity)
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  # decomp = list of tuples of the form (irrep, # of copies)
  print('Ecm_free: {:.6f}, degen: {}, irreps: {}'.format(Ecm_free, degen, decomp))
################################################################################
# Find solutions
################################################################################
# Create wrapper
func = lambda Ecm_arr: QC3(Ecm_arr[0], L, nnP, f_qcot_1sp, f_qcot_2s, M12,
                            waves, K3iso, K3B_par, K3E_par, parity, irrep)

# Iterate through levels & irreps
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  for (irrep, N_copies) in decomp:
    Ecm_test = Ecm_free + 0.06          # Try an energy slightly above Ecm_free
    Ecm_sol = fsolve(func, Ecm_test)[0] # Will only find one solution; more work needed if N_copies>1
    print('irrep: {}, Ecm_sol: {:.6f}, shift: {:.6f}'.format(irrep, Ecm_sol, Ecm_sol-Ecm_free))
