import numpy as np
from taurex.parameter import ParameterParser
from taurex.cache import OpacityCache, CIACache
from taurex.binning import SimpleBinner

# Set up caches
OpacityCache().set_opacity_path('/home/ubuntu/docs/files/xsecs')
CIACache().set_cia_path('/home/ubuntu/docs/files/cia/hitran')

# Load model
pp = ParameterParser()
pp.read('mascara_forward.par')
model = pp.generate_model()
model.build()

# Create coarse wavelength grid for binning
wl_bin = np.logspace(np.log10(0.6), np.log10(10.0), 1000)
wn_bin = np.sort(10000.0 / wl_bin)

# Run model and bin it
print("Running model and binning to 1000 points...")
binner = SimpleBinner(wngrid=wn_bin)
wngrid, rprs, _, _ = binner.bin_model(model.model(wngrid=wn_bin))

# Convert to wavelength
wavelength = 10000.0 / wngrid

# Save
output = np.column_stack([wavelength, rprs, np.sqrt(rprs)])
np.savetxt('../plots/mascara_forward.dat', output)

print(f"Done!!!!! with wavelength range: {wavelength.min():.3f} to {wavelength.max():.3f} microns")
print(f"Number of points: {len(wavelength)}")
