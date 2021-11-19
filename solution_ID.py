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
# Define ID system parameters (common mass rescaled to 1)
################################################################################
parity = -1         # Particle parity (-1 for pseudoscalars)
L = 5               # Box size (in units of 1/M1)
nnP = [0,0,0]       # 3-pt. FV spatial momentum (integer-valued)
################################################################################
# Define K2^{-1} parameters (s-wave only)
################################################################################
# Only s-wave dimers are used
a_s = 0.15   # s-wave scattering length
r_s = 0.     # s-wave effective range
################################################################################
# Define K2^{-1} phase shift function
################################################################################
f_qcot_s = [lambda q2: qcot_fits.qcot_fit_s(q2,[a_s,r_s],ERE=True)]
################################################################################
# Define Kdf3 parameters (isotropic only)
################################################################################
K3iso = [200, 400]      # Isotropic term is K3iso[0] + K3iso[1]*\Delta
################################################################################
# Define function that returns the smallest eigenvalue of QC in an irrep
################################################################################
def QC3_ID(Ecm, L, nnP, f_qcot_s, K3iso, parity, irrep):
  E = defns.E_to_Ecm(Ecm,L,nnP, rev=True)

  nnk_list = defns.list_nnk_nnP(E,L,nnP, Mijk=[1,1,1])
  F3 = F3_mat.F3mat_ID(E,L,nnP, f_qcot_s, nnk_list=nnk_list)
  K3 = K3main.K3mat_ID_iso(E,L,nnP, K3iso, nnk_list=nnk_list)

  F3i = LA.inv(F3)
  QC3_mat = F3i + K3
  QC3_mat_I = proj.irrep_proj_ID(QC3_mat,E,L,nnP,irrep,parity=parity)
  eigvals = sorted(defns.chop(LA.eigvals(QC3_mat_I).real,tol=1e-9), key=abs)
  return eigvals[0]
################################################################################
# Determine energy levels & irreps to include
################################################################################
Ecm_max = 5.   # This is the formal limit of validity of the QC3
#Ecm_max = 4.7   # We often explore higher energies though
print(Ecm_max)
# Find all free CM energy levels below Ecm_max, including degeneracies & irrep decomps
Ecm_free_decomp_list = GT.free_levels_decomp_3pt([1,1,1],L,nnP,Ecm_max=Ecm_max,sym='ID',parity=parity)
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  # decomp = list of tuples of the form (irrep, # of copies)
  print('Ecm_free: {:.6f}, degen: {}, irreps: {}'.format(Ecm_free, degen, decomp))
################################################################################
# Find solutions
################################################################################
# Create wrapper
func = lambda Ecm_arr: QC3_ID(Ecm_arr[0], L, nnP, f_qcot_s, K3iso, parity, irrep)

# Iterate through levels & irreps
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  for (irrep, N_copies) in decomp:
    Ecm_test = Ecm_free + 0.06          # Try an energy slightly above Ecm_free
    Ecm_sol = fsolve(func, Ecm_test)[0] # Will only find one solution; more work needed if N_copies>1
    print('irrep: {}, Ecm_sol: {:.6f}, shift: {:.6f}'.format(irrep, Ecm_sol, Ecm_sol-Ecm_free))
