################################################################################
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

# QC modules
import defns, group_theory_defns as GT, projections as proj
from F3 import F_mat, K2i_mat, qcot_fits
################################################################################

################################################################################
# Define 2-pt. system parameters --- *MUST ALWAYS RESCALE SO THAT M1=1*
################################################################################
sym = 'ND'
M1,M2 = [100.,50.]  # The 2-pt. system masses are [M1,M2], e.g. in MeV
M1,M2 = [1.,M2/M1]  # We always rescale by M1 to make everything dimensionless
M12 = [M1,M2]
#spec = 1           # spectator flavor: 1 for 12<->12 (ND) scattering, 2 for 11<->11 (ID) scattering
L = 5               # Box size (in units of 1/M1)
nnP = [0,0,0]       # 2-pt. FV spatial momentum (integer-valued)
if sym=='ID' and M12!=[1.,1.]:
  sys.error('Error: M12={} inconsistent with sym={}',M12,sym)
################################################################################
# Define K2^{-1} parameters
################################################################################
waves = 'sp'  # Partial waves used (only use s-wave for sym='ID')
a_s = 0.5    # s-wave scattering length
r_s = 0.2    # s-wave effective range
a_p = 0.3    # p-wave scattering length (only used if waves='sp' & sym='ND')
################################################################################
# Define K2^{-1} phase shift functions
################################################################################
f_qcot_s = lambda q2: qcot_fits.qcot_fit_s(q2,[a_s,r_s],ERE=True)
f_qcot_p = lambda q2: qcot_fits.qcot_fit_p(q2,a_p)
if waves=='s':
  f_qcot_waves = [f_qcot_s]
elif waves=='sp' and sym=='ND':
  f_qcot_waves = [f_qcot_s, f_qcot_p]
################################################################################
# Define function that returns the smallest eigenvalue of QC in an irrep
################################################################################
def QC2(Ecm, L, nnP, f_qcot_waves, M12, waves, irrep):
  M1,M2 = M12
  E = defns.E_to_Ecm(Ecm,L,nnP,rev=True)
  Pvec = 2*pi/L*np.array(nnP)
  F = 2*L**3 * F_mat.F_i_nnk(E+1,nnP,L,[0,0,0], Mijk=[1,M1,M2], waves=waves)
  K2i = 2* K2i_mat.K2i_inv_k(E+1,[0,0,0],Pvec,f_qcot_waves, Mijk=[1,M1,M2], waves=waves, eta_i=1, IPV=0)

  QC2_mat = defns.chop( F ) + defns.chop(K2i)
  P_I = proj.P_irrep_subspace_2pt(nnP,irrep,waves=waves)
  QC2_mat_I = defns.chop( P_I.T @ QC2_mat @ P_I )
  #QC2_mat_I = proj.irrep_proj_2pt(QC2_mat,nnP,irrep,waves=waves)
  eigvals = sorted( LA.eigvals(QC2_mat_I).real , key=abs)
  try:
    return eigvals[0]
  except:
    pass #print(Ecm,defns.hh(Ecm**2,Mjk=[M1,M2]),QC2_mat, QC2_mat_I, eigvals)
################################################################################
# Determine energy levels & irreps to include
################################################################################
#Ecm_max = (M1+M2) + 2*min(M1,M2) # This is the formal limit of validity of the QC2
Ecm_max = 3.5                     # We often explore higher energies though

# Find all free CM energy levels below Ecm_max, including degeneracies & irrep decomps
Ecm_free_decomp_list = GT.free_levels_decomp_2pt(M12,L,nnP,Ecm_max=Ecm_max,sym='ND',parity=1)
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  # decomp = list of tuples of the form (irrep, # of copies)
  print('Ecm_free: {:.6f}, degen: {}, irreps: {}'.format(Ecm_free, degen, decomp))
################################################################################
# Find solutions
################################################################################
# Create wrapper
func = lambda Ecm_arr: QC2(Ecm_arr[0], L, nnP, f_qcot_waves, M12, waves, irrep)

# Iterate through levels & irreps
for n in range(len(Ecm_free_decomp_list)):
  Ecm_free, degen, decomp = Ecm_free_decomp_list[n]
  for (irrep, N_copies) in decomp:
    if irrep == 'A1g' or (irrep=='T1u' and 'p' in waves):
      Ecm_test = Ecm_free + 0.06          # Try an energy slightly above Ecm_free
      Ecm_sol = fsolve(func, Ecm_test)[0] # Will only find one solution; more work needed if N_copies>1
    elif waves=='s':
      print('Warning: s-wave only affects A1g irrep;',
            'other irreps should not be considered in fit')
      Ecm_sol = Ecm_free
    elif waves=='sp':
      print("Warning: waves='sp' only affect A1g & T1u irreps;",
            'other irreps should not be considered in fit')
      Ecm_sol = Ecm_free
    print('irrep: {}, Ecm_sol: {:.6f}, shift: {:.6f}'.format(irrep, Ecm_sol, Ecm_sol-Ecm_free))
