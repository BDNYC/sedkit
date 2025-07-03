import pytest
from sedkit import sed
import astropy.units as u
import numpy as np


@pytest.fixture
def sub_spec():
    # Initialize the substellar SED
    sub_spec = sed.SED(name="2MASS J04151954-0935066", substellar=True)
    # Load in a spectrum to test
    sub_spec.add_spectrum_file(
        "tests/data/2MASS_J04151954-0935066_apparent_SED.txt",
        wave_units=u.micron,
        flux_units=u.erg / u.s / u.cm**2 / u.AA,
    )
    return sub_spec


@pytest.fixture
def spec():
    # Initialize the non_substellar SED
    spec = sed.SED(name="2MASS J04151954-0935066", substellar=False)
    # Load in a spectrum to test
    spec.add_spectrum_file(
        "tests/data/2MASS_J04151954-0935066_apparent_SED.txt",
        wave_units=u.micron,
        flux_units=u.erg / u.s / u.cm**2 / u.AA,
    )
    return spec


@pytest.mark.parametrize("seds", ["sub_spec", "spec"])
def test_just_spectrum(seds, request):
    sed = request.getfixturevalue(seds)
    sed.spectral_type = None    # forcing the SED to take no info from SIMBAD on SPT
    sed.distance = None     # forcing the SED to take no info from SIMBAD on distance

    sed.results
    assert np.isclose(sed.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(sed.fbol[1], 5.26164965e-15 * u.erg / u.s / u.cm**2)
    assert sed.mbol == (17.811, 0.003)
    assert sed.Lbol is None
    assert sed.Lbol_sun is None
    assert sed.radius is None
    assert sed.Teff is None
    assert sed.logg is None
    assert sed.mass is None


@pytest.mark.parametrize(
    "seds,radius_expected,teff_expected",
    [
        (
            "sub_spec",
            (0.973 * u.Rjup, 0.0 * u.Rjup, 0.0 * u.Rjup),
            (682 * u.K, 3.0 * u.K, 3.0 * u.K),
        ),
        (
            "spec",
            (0.1 * u.solRad, 0.0 * u.solRad, 0.0 * u.solRad),
            (682 * u.K, 3.0 * u.K, 3.0 * u.K),
        ),
    ],
)
def test_age_distance(seds, radius_expected, teff_expected, request):
    sed = request.getfixturevalue(seds)
    sed.age = 4.5 * u.Gyr, 4 * u.Gyr
    sed.parallax = 175.2 * u.mas, 1.7 * u.mas     # This is the parallax of 2MASS J04151954-0935066
    sed.results

    assert np.isclose(sed.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(sed.fbol[1], 5.51305904e-15 * u.erg / u.s / u.cm**2)
    assert sed.mbol == (17.811, 0.003)
    assert np.isclose(sed.Lbol[0], 7.48405489e+27 * u.erg / u.s, rtol=0.5)
    assert np.isclose(sed.Lbol[1], 1.32712721e+26 * u.erg / u.s, rtol=0.5)
    assert np.isclose(sed.Lbol_sun[0], -5.709, rtol=0.2)
    assert np.isclose(sed.Lbol_sun[1], 0.009, rtol=0.2)
    assert sed.radius == radius_expected
    assert sed.Teff == teff_expected
    assert sed.logg is None
    assert sed.mass is None


@pytest.mark.parametrize(
    "seds,radius_expected",
    [
        ("sub_spec",
         [0.855 * u.Rjup, 0.023 * u.Rjup, 0.112 * u.Rjup]
         ),
        ("spec",
         [0.088 * u.solRad, 0.002 * u.solRad, 0.012 * u.solRad]
         ),
    ],
)
def test_radius_using_evo_model(seds, radius_expected, request):
    sed = request.getfixturevalue(seds)
    sed.age = 4.5 * u.Gyr, 4 * u.Gyr
    sed.parallax = 175.2 * u.mas, 1.7 * u.mas
    sed.evo_model = "hybrid_solar_age"  # Saumon & Marley 2008 evo model
    sed.results
    sed.infer_radius(infer_from="evo_model")
    assert np.isclose(sed.radius[0], radius_expected[0],rtol=0.2)
    assert np.isclose(sed.radius[1], radius_expected[1], rtol=0.2)
    assert np.isclose(sed.radius[2], radius_expected[2], rtol=0.2)

@pytest.mark.parametrize(
    "seds,radius_expected",
    [
        ("sub_spec",
         [0.86 * u.Rjup, 0.001 * u.Rjup, 0.001 * u.Rjup]
         ),
        ("spec",
         [0.088 * u.solRad, 0.0 * u.solRad, 0.0 * u.solRad]
         ),
    ],
)
def test_age_distribution(seds,radius_expected,request):
    sed = request.getfixturevalue(seds)
    sed.age = 4.5 * u.Gyr, 0.1 * u.Gyr
    sed.parallax = 175.2 * u.mas, 1.7 * u.mas
    sed.evo_model = "hybrid_solar_age"  # Saumon & Marley 2008 evo model
    sed.age_distribution = 'normal'
    sed.results
    sed.infer_radius(infer_from="evo_model")
    assert sed.radius == radius_expected