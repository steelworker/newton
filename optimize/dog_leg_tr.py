# import numpy as np
import torch as tr
import numpy.linalg as ln
import scipy as sp
from math import sqrt

# Objective function    
def f(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    
# Gradient
def jac(x):
    return [-400*(x[1] - x[0]**2)*x[0] - 2 + 2*x[0], 200*x[1] - 200*x[0]**2]
# Hessian
def hess(x):
    return [[1200*x[0]**2 - 400*x[1]+2, -400*x[0]], [-400*x[0], 200]]

    
def dogleg_method(Hk, gk, Bk, trust_radius):

    # Compute the Newton point.
    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    pB = -Hk @ gk
    norm_pB = sqrt(pB @ pB)

    # Test if the full step is within the trust region.
    if norm_pB <= trust_radius:
        return pB
    
    # Compute the Cauchy point.
    # This is the predicted optimum along the direction of steepest descent.
    pU = - (gk @ gk) / gk @ (Bk @ gk) * gk
    dot_pU = pU @ pU
    norm_pU = sqrt(dot_pU)

    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    # Find the solution to the scalar quadratic equation.
    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + tau*(p_b - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    pB_pU = pB - pU
    dot_pB_pU = pB_pU @ pB_pU
    dot_pU_pB_pU = pU @ pB_pU
    fact = dot_pU_pB_pU**2 - dot_pB_pU * (dot_pU - trust_radius**2)
    tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU
    
    # Decide on which part of the trajectory to take.
    return pU + tau * pB_pU
    

def trust_region_dogleg(func, jac, hess, x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, gtol=1e-4, 
                        maxiter=100):
    xk = x0
    trust_radius = initial_trust_radius
    k = 0
    while True:
        print(k)
      
        gk = jac(xk)
        Bk = hess(xk)
        Hk = tr.linalg.inv(Bk)
        
        pk = dogleg_method(Hk, gk, Bk, trust_radius)
       
        # Actual reduction.
        act_red = func(xk) - func(xk + pk)
        
        # Predicted reduction.
        pred_red = -((gk @ pk) + 0.5 * (pk @ (Bk @ pk)))
        
        # Rho.
        rhok = act_red / pred_red
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red
            
        # Calculate the Euclidean norm of pk.
        norm_pk = sqrt(pk @ pk)
        
        # Rho is close to zero or negative, therefore the trust region is shrunk.
        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else: 
        # Rho is close to one and pk has reached the boundary of the trust region, therefore the trust region is expanded.
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0*trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius
        
        # Choose the position for the next iteration.
        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk
            
        # Check if the gradient is small enough to stop
        if tr.linalg.norm(gk) < gtol:
            break
        
        # Check if we have looked at enough iterations
        if k >= maxiter:
            break
        k = k + 1
    return xk