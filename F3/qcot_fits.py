import numpy as np, sys
sqrt = np.sqrt; pi=np.pi


# s-wave phase shift model (for K2)
#@njit(fastmath=True)
def qcot_fit_s(q2,par_vec,ERE=True): # Use effective range expansion by default
    if ERE==True:
      a0 = par_vec[0]
      qcot = -1/a0
      if len(par_vec)==2:
        r = par_vec[1]
        qcot += 1/2*r*q2
      elif len(par_vec)>2:
        sys.error("Error in qcot_fit_s: Too many parameters in par_vec; expected 1 or 2")
        raise IndexError
      return qcot
    # else:
    #   s = 4.0*(1.0+q2)
    #   qcot = sqrt(s)/(s-2) * np.dot(par_vec,np.array([q2**i for i in range(len(par_vec))]))
    #   return qcot


# p-wave phase shift model (for K2)
def qcot_fit_p(q2,a1):
  return -1/a1
  # s = 4.0*(1.0+q2)
  # return -1 + a2*(s/4)**(3/2)
