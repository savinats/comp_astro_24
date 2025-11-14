#!/usr/bin/env python
# coding: utf-8

# In[4]:


import batman
import numpy as np
import matplotlib.pyplot as plt

#exoplanet MASCARA-3 Ab

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 2457146.6                 #time of inferior conjunction
params.per = 5.5514926                #orbital period
params.rp = 0.084467                  #planet radius (in units of stellar radii)
params.a = 9.90312                    #semi-major axis (in units of stellar radii)
params.inc = 89.16                    #orbital inclination (in degrees)
params.ecc = 0.085                    #eccentricity
params.w = 41.                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.215, 0.29]              #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(0, 1, 500)            #times at which to calculate light curve
m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)          #calculates light curve

new_flux = m.light_curve(params)

plt.figure(figsize=(10, 6))
plt.plot(t, flux, label='Transit Light Curve')
plt.xlabel('Time (days)')
plt.ylabel('Relative Flux')
plt.title('Transit Light Curve of Mascara-3 Ab')
plt.legend()
plt.grid()
plt.show()