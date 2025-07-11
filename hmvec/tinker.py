import numpy as np
from scipy.interpolate import interp1d
import os

"""
Implements Tinker et al 2010 and Tinker et al 2008

nu = deltac / sigma


nu and sigma have shape
(numzs,numms)

So a function that asks for redshifts zs in additions
expects consistent number of redshifts.
"""

constants = {
    'deltac': 1.686,
}

default_params = {
    'tinker_f_nu_alpha_z0_delta_200':0.368, # Tinker et al 2010 Table 4
    }

def bias(nu,delta=200.):
    # Eq 6 of Tinker 2010
    deltac = constants['deltac']
    y = np.log10(delta)
    A = lambda y: 1. + 0.24*y*np.exp(-(4./y)**4.)
    a = lambda y: 0.44*y-0.88
    B = 0.183
    b = 1.5
    C = lambda y: 0.019 + 0.107*y + 0.19 *np.exp(-(4./y)**4.)
    c = 2.4
    nua = nu**a(y)
    t1 = (nua)/(nua+deltac**a(y))
    t2 = nu**b
    t3 = nu**c
    return 1 - A(y)*t1 + B*t2 + C(y)*t3

    
def f_nu(nu,zs,delta=200.,norm_consistency=True,
         alpha=default_params['tinker_f_nu_alpha_z0_delta_200']):
    # This is the f of Tinker 2010
    # but not the f of Tinker 2008
    # Tinker 2008 f = g in Appendix = nu * f of Tinker 2010
    # \int b f dnu should be 1 (in fact, norm_consistency enforces this for z>0)
    # This should be equiavelnt to \int dm (m/rho) n b = 1 (bias consistency)
    # if n = (rho/m) nu f(nu) dlnsigmainv/dm
    assert np.isclose(delta,200.), "delta!=200 note implemented yet." # FIXME: interpolate for any value of delta
    # FIXME: set z>3 to z=3
    # zs = zs*np.heaviside(3-zs,0)+3*np.heaviside(zs-3,0)
    zs = np.where(zs < 3, zs, 3)
    beta0 = 0.589
    gamma0 = 0.864
    phi0 = -0.729
    eta0 = -0.243
    beta  = beta0  * (1+zs)**(0.20)
    phi   = phi0   * (1+zs)**(-0.08)
    eta   = eta0   * (1+zs)**(0.27)
    gamma = gamma0 * (1+zs)**(-0.01)
    unnormalized = (1. + (beta*nu)**(-2.*phi))*(nu**(2*eta))*np.exp(-gamma*nu**2./2.)
    if norm_consistency:
        aroot = os.path.dirname(__file__)+"/data/alpha_consistency.txt"
        izs,ialphas = np.loadtxt(aroot,unpack=True) # FIXME: hardcoded
        alpha = interp1d(izs,ialphas,bounds_error=True)(zs)
    return alpha * unnormalized 

A_array = np.array([1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01, 2.480968e-01,
        2.546053e-01, 2.600000e-01, 2.600000e-01, 2.600000e-01])
a_array = np.array([1.466904, 1.521782, 1.559186, 1.614585, 1.869936,
        2.128056, 2.301275, 2.529241, 2.661983])
b_array = np.array([2.571104, 2.254217, 2.048674, 1.869559, 1.588649,
        1.507134, 1.464374, 1.436827, 1.405210])
c_array = np.array([1.193958, 1.270316, 1.335191, 1.446266, 1.581345,
        1.795050, 1.965613, 2.237466, 2.439729])
delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

A_interp = interp1d(delta_virs, A_array,)
a_interp = interp1d(delta_virs, a_array,)
b_interp = interp1d(delta_virs, b_array,)
c_interp = interp1d(delta_virs, c_array,)
    
def simple_f_nu(nu, zs, omm):
    delta = 200#/omm
    # assert np.isclose(delta,200.), "delta!=200 not implemented yet." # FIXME: interpolate for any value of delta
    deltac = constants['deltac']
    sigma = deltac/nu
    if delta == 200:
        A0 = 0.186
        a0 = 1.47
        b0 = 2.57
        c = 1.19
    else:
        A0 = 0.1 * np.log10(delta) - 0.05
        a0 = 1.43 + (np.log10(delta) - 2.3)**1.5
        b0 = 1.0 + (np.log10(delta) - 1.6)**-1.5
        c = 1.2 + (np.log10(delta) - 2.35)**1.6
    alpha = 10 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
    A = A0 * (1+zs)**(-0.14)
    a = a0 * (1+zs)**(-0.06)
    b = b0 * (1+zs)**(-alpha)
    return A* (1.+((sigma/b)**(-a))) * np.exp(-c/sigma**2.)


def NlnMsub(Msubs,Mhosts):
    """
    Eq 12 of the *published* version of J. L. Tinker and A. R. Wetzel, apj 719, 88 (2010),
    0909.1325
    Differs from arxiv in the 0.3 prefactor
    Accepts 1d array of Msubs and Mhosts
    and returns 2d array for (Msubs,Mhosts)
    """
    mrat = Msubs[:,None]/Mhosts[None,:]
    return 0.3 * (mrat**(-0.7)) * np.exp(-9.9 * (mrat**2.5))
