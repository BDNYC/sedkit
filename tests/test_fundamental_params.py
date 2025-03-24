import pytest
from sedkit import sed
import astropy.units as u
import numpy as np

@pytest.fixture
def spec():
    # Load in a spectrum to test
    spec = sed.SED(name = "2MASS J04151954-0935066")
    spec.add_spectrum_file(
        "tests/data/2MASS_J04151954-0935066_apparent_SED.txt",
        wave_units=u.micron,
        flux_units=u.erg / u.s / u.cm**2 / u.AA,
    )

    # Calculate the fundamental parameters using default settings
    spec.results
    return spec


def test_just_spectrum(spec):
    assert np.isclose(spec.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(spec.fbol[1], 5.26164965e-15 * u.erg / u.s / u.cm**2)
    assert spec.mbol[0] == 17.811
    assert spec.mbol[1] == 0.003
    assert spec.Lbol is None
    assert spec.radius is None
    assert spec.Teff is None
    assert spec.logg is None


def test_age_distance(spec):
    spec.age = 4.5 * u.Gyr, 0.1 * u.Gyr
    spec.distance = 10 * u.pc, 0.1 * u.pc
    spec.results

    assert np.isclose(spec.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(spec.fbol[1], 5.26164965e-15 * u.erg / u.s / u.cm**2)
    assert spec.mbol[0] == 17.811 
    assert spec.mbol[1] == 0.003
    assert np.isclose(spec.Lbol[0], 2.29543367e+28 * u.erg / u.s)
    assert np.isclose(spec.Lbol[1], 4.63561547e+26  * u.erg / u.s, rtol=0.05)
    assert spec.Teff == (903 * u.K,  4.0 * u.K, 4.0 * u.K)
    assert spec.radius == (0.1 * u.Rsun, 0.0 * u.Rsun, 0.0 * u.Rsun)
    assert spec.logg is None


def test_radius(spec):
     
    spec.infer_radius(infer_from='evo_model')
