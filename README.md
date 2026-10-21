# gaiamockplus

**gaiamockplus** extends [gaiamock](https://github.com/kareemelbadry/gaiamock), a fast forward model of Gaia epoch astrometry, to stars hosting any number of planets (or other dark companions). It can:

- simulate Gaia epoch astrometry (and RVs) of N-planet systems;
- fit them: astrometry, RVs from multiple instruments, or both jointly, searching for planets with C-based optimizers for a known number of planets or blindly;
- provide likelihoods and starting points for MCMC samplers.

All of gaiamock is included (`from gaiamockplus import gaiamock`). If you use gaiamockplus, please cite [El-Badry et al. (2024)](https://arxiv.org/abs/2411.00088).

![Best-fit photocenter orbit and along-scan angle residuals for a simulated single-planet system](figures/example_fit.png)
*Simulated Gaia DR5 astrometry of a star with an 8 Jupiter-mass planet ([notebook 01](notebooks/01_single_planet_astrometry.ipynb)): the best-fit photocenter orbit (left), and the along-scan residuals of the best-fit models without and with the planet (right).*

## Installation

```bash
pip install git+https://github.com/CalebLammers/gaiamockplus.git
```

This compiles the package's C code, which needs a C compiler and the [GSL](https://www.gnu.org/software/gsl/) library (e.g. `brew install gsl` on macOS, or `sudo apt install libgsl-dev` on Debian/Ubuntu). macOS and Linux are supported. For development, clone the repository and run `pip install -e .`.

The first time Gaia's scanning law is needed, gaiamockplus downloads it (286 MB; 984 MB unzipped) to `~/.gaiamockplus`. To use an existing copy of gaiamock's `healpix_scans/` folder instead, set the environment variable `GAIAMOCKPLUS_DATA` to the folder that contains it.

The example notebooks also need `emcee` and `corner` (`pip install emcee corner`).

## Examples

- [01_single_planet_astrometry](notebooks/01_single_planet_astrometry.ipynb): simulate and fit the astrometry of a single-planet system, and sample the posterior with MCMC.
- [02_two_planet_astrometry](notebooks/02_two_planet_astrometry.ipynb): a blind search for astrometric planets in a two-planet system.
- [03_single_planet_astrometry_and_rvs](notebooks/03_single_planet_astrometry_and_rvs.ipynb): RVs from three instruments, fit alone and jointly with the astrometry.
- [04_two_planet_astrometry_and_rvs](notebooks/04_two_planet_astrometry_and_rvs.ipynb): two planets with astrometry and RVs, and their mutual inclination.

## Contact

Questions/requests can be sent to caleb [dot] lammers [at] princeton [dot] edu.
