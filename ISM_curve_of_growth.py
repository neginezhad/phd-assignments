import numpy as np
import matplotlib.pyplot as plt
import pint
from scipy.special import voigt_profile

# Initialize unit registry for handling units
ureg        = pint.UnitRegistry()
charge_e    = 4.8032e-10 * ureg.esu  # elementary charge
mass_e      = ureg.electron_mass  # mass of an electron
speed_light = ureg.speed_of_light  # speed of light constant

plt.figure(figsize=[10, 20])

def compute_voigt(v_velocity, doppler_b, gamma_damp, wavelength):
    """
    Returns the Voigt profile given velocity, doppler parameter, damping coefficient, and wavelength.
    Units: v_velocity (km/s), doppler_b (km/s), gamma_damp (1/s), wavelength (angstrom)
    """
    gamma_damp = gamma_damp / ureg.s
    doppler_b  = doppler_b * ureg.km / ureg.s
    wavelength = wavelength * ureg.angstrom

    # Voigt parameter computation
    damp_factor = gamma_damp / (4 * np.pi) * wavelength
    sigma_width = doppler_b / np.sqrt(2)

    # Convert parameters to base units for calculation
    sigma_width = sigma_width.to('km/s').magnitude
    damp_factor = damp_factor.to('km/s').magnitude

    # Voigt profile result
    return voigt_profile(v_velocity, sigma_width, damp_factor) * ureg.s / ureg.km

def validate_voigt():
    # Testing for Voigt profile integration close to unity
    velocity_range   = np.arange(-300000, 300000, 0.1)
    doppler_param    = 10
    gamma_param      = 100
    wavelength_param = 1000
    voigt_values     = compute_voigt(velocity_range, doppler_param, gamma_param, wavelength_param).magnitude

    assert np.allclose(np.trapz(voigt_values, velocity_range), 1, rtol=1e-4), \
        f"Expected Voigt integral to approximate 1 but got {np.trapz(voigt_values, velocity_range)}"

def optical_depth(v_velocity, column_density, osc_strength, wavelength, doppler_b, gamma_damp):
    """
    Computes the optical depth profile for given physical parameters.
    Units: column_density (cm^-2), osc_strength (dimensionless), wavelength (angstrom), doppler_b (km/s)
    """
    phi_velocity   = compute_voigt(v_velocity, doppler_b, gamma_damp, wavelength)
    column_density = column_density * ureg.cm ** -2
    wavelength     = wavelength * ureg.angstrom
    doppler_b      = doppler_b * ureg.km / ureg.s
    prefactor      = np.pi * charge_e ** 2 / (mass_e * speed_light) * column_density * osc_strength * wavelength 
    tau_values     = prefactor * phi_velocity

    tau_values = tau_values.to_base_units()
    assert tau_values.units == ureg.dimensionless, f"Unexpected units in tau_values: {tau_values.units}"
    
    return tau_values.magnitude

def compute_equivalent_width(column_density, osc_strength, wavelength, doppler_b, gamma_damp, approx=False, v_range=300000):
    """
    Calculates equivalent width based on parameters. If approx=True, uses approximation formula.
    """
    if approx: 
        tau_zero   = optical_depth(0, column_density, osc_strength, wavelength, doppler_b, gamma_damp)
        doppler_b  = doppler_b * ureg.km / ureg.s
        gamma_damp = gamma_damp / ureg.s
        wavelength = wavelength * ureg.angstrom

        if tau_zero < 1.25393:
            width_total = np.sqrt(np.pi) * doppler_b / speed_light * tau_zero / (1 + tau_zero / (2 * np.sqrt(2))) 
        else:
            width_total = np.sqrt((2 * doppler_b / speed_light) ** 2 * np.log(tau_zero / np.log(2)) + \
                                  (doppler_b / speed_light) * (gamma_damp * wavelength / speed_light) * ((tau_zero - 1.25393) / np.sqrt(np.pi)))
            width_total = width_total.to('dimensionless')
        width_total     = width_total * wavelength
        width_total     = width_total.to('angstrom').magnitude

    else:
        velocity_grid = np.arange(-v_range, v_range, 0.1)
        tau_vals      = optical_depth(velocity_grid, column_density, osc_strength, wavelength, doppler_b, gamma_damp)
        width_total   = np.trapz(1 - np.exp(-tau_vals), velocity_grid) * ureg.km / ureg.s
        width_total   = width_total * wavelength * ureg.angstrom / ureg.c
        width_total   = width_total.to('angstrom').magnitude

    return width_total

def plot_curve_growth(column_densities, osc_strength, wavelength, doppler_b_values, gamma_damp, approx=False, v_range=300000, title=''):
    for i, doppler_b in enumerate(doppler_b_values):
        log_W_values, log_N_values = [], []
        for col_density in column_densities:
            total_width = compute_equivalent_width(col_density, osc_strength, wavelength, doppler_b, gamma_damp, approx=approx, v_range=v_range)
            log_W_values.append(np.log10(total_width))
            log_N_values.append(np.log10(col_density))

        plt.plot(log_N_values, log_W_values, '-', label=f"b = {doppler_b:.1f} km/s")

    plt.title("Growth Curve for " + title)
    plt.xlabel("log(N [cm$^{-2}$])")
    plt.ylabel("log(W$_{\lambda}$ [Å])")
    plt.legend()
    plt.grid()

if __name__ == '__main__':
    # Iron (Fe II) transition at 2382.8 Å
    is_approx     = True
    osc_strength  = 0.320
    wavelength_fe = 2382.7642
    damping_gamma = 3.13e8
    redshift_z    = 2
    W_fe_approx   = compute_equivalent_width(10**20.3, osc_strength, wavelength_fe, 5, damping_gamma, approx=is_approx)
    print(f"W_rest (Fe II) = {W_fe_approx:.3f} Å")

    doppler_b_vals      = np.array([1, 2, 3, 5, 10], dtype=float)
    column_density_vals = 10 ** np.arange(12, 17, 0.2, dtype=float)
    plt.subplot(3, 1, 1)
    plot_curve_growth(column_density_vals, osc_strength, wavelength_fe, doppler_b_vals, damping_gamma, approx=is_approx, title='Fe II 2382.8')
    plt.xlim(12,16)

    # Iron (Fe II) transition at 2249.9 Å
    osc_strength  = 0.00182
    wavelength_fe = 2249.8768
    damping_gamma = 3.31e8
    W_fe_approx   = compute_equivalent_width(10**20.3, osc_strength, wavelength_fe, 5, damping_gamma, approx=is_approx)
    print(f"W_rest (Fe II) = {W_fe_approx:.3f} Å")

    plt.subplot(3, 1, 2)
    plot_curve_growth(column_density_vals, osc_strength, wavelength_fe, doppler_b_vals, damping_gamma, approx=is_approx, title='Fe II 2249.9')
    plt.xlim(12,16)

    # Carbon (C II) transition at 1334.5 Å
    osc_strength  = 0.12780
    wavelength_c  = 1334.5323
    damping_gamma = 2.880e8
    W_c_approx    = compute_equivalent_width(10**20.3, osc_strength, wavelength_c, 5, damping_gamma, approx=is_approx)
    print(f"W_rest (C II) = {W_c_approx:.3f} Å")

    plt.subplot(3, 1, 3)
    plot_curve_growth(column_density_vals, osc_strength, wavelength_c, doppler_b_vals, damping_gamma, approx=is_approx, title='C II 1334.5')
    plt.xlim(13,17)
    plt.ylim(-2,)

    plt.savefig('curve_of_growth.png')