import numpy as np
import taurex.log
taurex.log.disableLogging()

from taurex.cache import OpacityCache, CIACache
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Isothermal
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.pressure import SimplePressureProfile
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.binning import SimpleBinner
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer

class ForwardModelRunner:

    def __init__(self, config):
        self.config = config

    def run(self):
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(self.config['paths']['xsec_path'])
        CIACache().set_cia_path(self.config['paths']['cia_path'])

        planet = Planet(
            planet_radius=self.config['planet']['radius'],
            planet_mass=self.config['planet']['mass']
        )

        star = BlackbodyStar(
            temperature=self.config['star']['temperature'],
            radius=self.config['star']['radius']
        )
        
        pconf = self.config['pressure']

        pressure = SimplePressureProfile(
            atm_min_pressure=float(pconf['atm_min_pressure']),
            atm_max_pressure=float(pconf['atm_max_pressure']),
            nlayers=int(pconf['nlayers'])
        )

        temperature = Isothermal(
            T=self.config['temperature']['T']
        )

        chemconf = self.config['chemistry']

        chemistry = TaurexChemistry(
            fill_gases=chemconf['fill_gases'],
            ratio=float(chemconf['ratio'])
        )

        for gas, mr in chemconf['gases'].items():
                chemistry.addGas(ConstantGas(gas, float(mr)))


        model = TransmissionModel(
            planet=planet,
            star=star,
            temperature_profile=temperature,
            chemistry=chemistry,
            pressure_profile=pressure
        )

        model.add_contribution(AbsorptionContribution())
        model.add_contribution(CIAContribution(cia_pairs=['H2-H2', 'H2-He']))
        model.add_contribution(RayleighContribution())

        model.build()

        wngrid = np.sort(10000 / np.logspace(-0.4, 1.1, 1000))
        wl, spec, _, _ = model.model(wngrid=wngrid)

        binner = SimpleBinner(wngrid=wngrid)
        bin_wn, bin_spec, _, _ = binner.bin_model((wl, spec))

        wavelength = 10000.0 / bin_wn
        error = np.full_like(bin_spec, 1e-5)

        output = np.column_stack([wavelength, bin_spec, error])
        np.savetxt(
            self.config['output']['spectrum'],
            output,
            header="Wavelength(um) (Rp/Rs)^2 Error"
        )

        print(f"Forward model saved to {self.config['output']['spectrum']}")
        
class RetrievalModelRunner:

    def __init__(self, config):
        self.config = config

    def run(self):
        OpacityCache().set_opacity_path(self.config["paths"]["xsec_path"])
        CIACache().set_cia_path(self.config["paths"]["cia_path"])

        planet = Planet(
            planet_radius=self.config["planet"]["radius"],
            planet_mass=self.config["planet"]["mass"]
        )
        star = BlackbodyStar(
            temperature=self.config["star"]["temperature"],
            radius=self.config["star"].get("radius", 1)
        )

        pressure = SimplePressureProfile(
            atm_min_pressure=float(self.config["pressure"]["atm_min_pressure"]),
            atm_max_pressure=float(self.config["pressure"]["atm_max_pressure"]),
            nlayers=int(self.config["pressure"]["nlayers"])
        )
        temperature = Isothermal(T=self.config["temperature"]["T"])
        temperature.nlayers = self.config["pressure"]["nlayers"]

        chemistry = TaurexChemistry(
            fill_gases=self.config["chemistry"]["fill_gases"],
            ratio=self.config["chemistry"]["ratio"]
        )
        
        for gas, val in self.config["chemistry"]["gases"].items():
            if val is not None:
                chemistry.addGas(ConstantGas(gas, float(val)))

        model = TransmissionModel(
            planet=planet,
            star=star,
            temperature_profile=temperature,
            chemistry=chemistry,
            pressure_profile=pressure
        )
        for contrib in self.config["model"]["contributions"]:
            if contrib.lower() == "absorption":
                model.add_contribution(AbsorptionContribution())
            elif contrib.lower() == "rayleigh":
                model.add_contribution(RayleighContribution())
            elif contrib.lower() == "cia":
                model.add_contribution(CIAContribution())
        model.build()
        
        model.chemistry.initialize_chemistry(
            nlayers=model.nLayers,
            temperature_profile=model.temperatureProfile,
            pressure_profile=model.pressureProfile,
            altitude_profile=None
        )

        obs = ObservedSpectrum(self.config["retrieval"]["input_spectrum"])

        optimizer = NestleOptimizer()
        optimizer.set_model(model)
        optimizer.set_observed(obs)

        free_params = self.config["retrieval"]["free_parameters"]
        for param_name, param_info in free_params.items():
            optimizer.enable_fit(param_name)

            min_val = float(param_info.get("min"))
            max_val = float(param_info.get("max"))

            if min_val is None or max_val is None:
                raise ValueError(f"Parameter '{param_name}' must have 'min' and 'max' defined!")

            prior_type = param_info.get("prior_type", "uniform").lower()

            if prior_type == "log-uniform":
                if min_val <= 0 or max_val <= 0:
                    raise ValueError(f"Log-uniform prior for '{param_name}' requires positive min/max values.")
    
            optimizer.set_boundary(param_name, [min_val, max_val])

        print("Starting retrieval...")
        optimizer.fit()
        print("Retrieval complete!")

        wl, spec, _, _ = model.model()
        np.savetxt(
            self.config["retrieval"]["output_spectrum"],
            np.column_stack([wl, spec]),
            header="Wavelength (micron)  (Rp/Rs)^2"
        )
        print(f"Retrieved spectrum saved to {self.config['retrieval']['output_spectrum']}")