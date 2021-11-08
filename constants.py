
################################################################################
# Spectator cutoff constants
################################################################################
# Set xmin, xmax for J cutoff fn.
def get_xrange():
  xmin = 0.02
  xmax = 0.97
  return xmin, xmax

# Set epsH for H cutoff fn.
def get_epsH():
  epsH = 0
  return epsH

################################################################################
# UV cutoff constants in F
################################################################################
# Set alpha_KSS (for smooth exponential UV cutoff)
def get_alpha():
  alpha = 0.5
  return alpha

# Set minimum cutoff for integral of UV momenta (R_Lambda=epsilon in beyond_iso)
def get_cutoff():
  cutoff = 1e-9
  return cutoff
