################################################################################
# Import relevant python modules
################################################################################
# Standard python modules
import numpy as np, sys, os
pi = np.pi; sqrt=np.sqrt; LA=np.linalg

# Set up paths for loading QC modules
from pathlib import Path
cwd = Path(os.getcwd())
sys.path.insert(0,str(cwd/'F3'))
sys.path.insert(0,str(cwd/'Kdf3'))

# QC modules
import defns, group_theory_defns as GT, projections as proj
from F3 import G_mov, F_fast, K2i_mat, qcot_fits, F3_mat
from Kdf3 import K3main
################################################################################

################################################################################
# Define 2+1 system parameters --- *ALWAYS SET M1=1*
################################################################################
M1,M2 = [1.,0.5]; M12 = [M1,M2] # The 3-pt. system masses are [M1,M1,M2]
                                # *NOTE*: WE ALWAYS RESCALE BY M1, SO SET M1=1
parity = -1                     # Particle parity (-1 for pseudoscalars, +1 for scalars)
L = 5                           # Box size (in units of 1/M1)
nnP = [1,2,0]                   # 3-pt. FV spatial momentum (integer-valued)
E = 3.7                         # Total 3-pt. energy in moving frame (in units of M1)

Ecm = defns.E_to_Ecm(E,L,nnP)   # Total CM 3-pt. energy (in units of M1)
Pvec = 2*pi/L*np.array(nnP)     # 3-pt. spatial momentum (in units of M1)
# Ecm = 2*M1+M2 + 0.5*M1
# E = defns.E_to_Ecm(Ecm,L,nnP,rev=True)
################################################################################
# Define K2^{-1} parameters
################################################################################
waves = 'sp'            # Partial waves used for dimers with flavor-1 spectators
                        # (flavor-2 spectators only use s-wave)
a_1s, r_1s = [0.5, 0.]  # s-wave scattering length & effective range for spectator-flavor-1 channel
a_1p = 0.3              # p-wave scattering length for spectator-flavor-1 channel
a_2s = 0.4              # s-wave scattering length for spectator-flavor-2 channel
################################################################################
# Define Kdf3 parameters
################################################################################
K3iso = [200, 400]      # Isotropic term is K3iso[0] + K3iso[1]*\Delta
K3B_par = 5000          # Parameter for K3B term
K3E_par = 3000          # Parameter for K3E term
################################################################################
# Define K2^{-1} phase shift functions
################################################################################
f_qcot_1s = lambda q2: qcot_fits.qcot_fit_s(q2,[a_1s,r_1s],ERE=True)
f_qcot_1p = lambda q2: qcot_fits.qcot_fit_p(q2,a_1p)
f_qcot_1sp = [f_qcot_1s, f_qcot_1p]
f_qcot_2s = [lambda q2: qcot_fits.qcot_fit_s(q2,[a_2s],ERE=True)]
################################################################################
# Determine relevant flavor-1 and flavor-2 spectator momenta
################################################################################
nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
nnk_lists_12 = [nnk_list_1, nnk_list_2]
print('flavor 1 spectators:',nnk_list_1, '\nflavor 2 spectators:', nnk_list_2,'\n')
################################################################################
# Compute desired QC matrices
################################################################################
# F = F_fast.F_full_2plus1_scratch(E,nnP,L, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, diag_only=False)
# K2i = K2i_mat.K2_inv_mat_2plus1(E,L,nnP,f_qcot_1sp,f_qcot_2s, M12=M12, waves=waves, nnk_lists_12=nnk_lists_12, IPV=0)
# G = G_mov.Gmat_2plus1(E,L,nnP, M12=M12, nnk_lists_12=None, waves=waves)
# J =  defns.chop( 1/3*np.eye(len(F)) - LA.inv(K2i + F + G) @ F )
# F3 = F @ J
F3 = F3_mat.F3mat_2plus1(E,L,nnP, f_qcot_1sp,f_qcot_2s, M12=M12,waves=waves,nnk_lists_12=nnk_lists_12)
K3 = K3main.K3mat_2plus1(E,L,nnP, K3iso,K3B_par,K3E_par, M12=M12,waves=waves,nnk_lists_12=nnk_lists_12)
F3i = LA.inv(F3)

QC3_mat = F3i + K3            # QC3 matrix as defined in the paper
#QC3_mat = LA.inv(J) + K3@F   # This works better than F3^{-1} + Kdf3; removes spurious free solutions
#QC2_mat = F + K2i            # This works better than F^{-1} + K2; removes spurious free solutions
################################################################################
# Perform desired irrep projections
################################################################################
for name,M in [('QC3_mat',QC3_mat)]: #[('F',F),('G',G),('K2i',K2i),('K3',K3),('F3i',F3i),('QC3_mat',QC3_mat)]:
  print(name, 'size:', len(M))
  for I in GT.irrep_list(nnP):
    M_I = proj.irrep_proj_2plus1(M,E,L,nnP,I, M12=M12, waves=waves, parity=parity)
    if M_I.shape != (0,0):
      M_I_eigs = sorted(defns.chop(LA.eigvals(M_I).real,tol=1e-12), reverse=True,key=abs)
      print(I, M_I_eigs)
  print()
