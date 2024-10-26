import batman
import numpy as np
import matplotlib.pyplot as plt

#exoplanet MASCARA-3 Ab

#Define parameters oof exoplanet
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 2457146.6                 #time of inferior conjunction
params.per = 5.5514926                #orbital period
params.rp = 0.1278                    #planet radius (in units of stellar radii) after converting it from jupiter radii 
params.a = 0.06971                    #semi-major axis (in units of stellar radii)
params.inc = 89.16                    #orbital inclination (in degrees)
params.ecc = 0.085                    #eccentricity
params.w = 41.                        #longitude of periastron (in degrees)

#Here i assume that the time of the first transit is the time of inferior conjuction
#I converted the planet radius to stellar radii bc the encyclopedia had it in jupiter radii  

t = np.linspace(2400000, t0, 2500000)  #times at which to calculate light curve
m = batman.TransitModel(params, t)     #initializes model

flux = m.light_curve(params)           #calculates light curve

#plot the transit
plt.figure(figsize=(10, 6))
plt.plot(t, flux, label='Transit Light Curve')
plt.xlabel('Time (days)')
plt.ylabel('Relative Flux')
plt.title('Transit Light Curve')
plt.legend()
plt.grid()
plt.show()