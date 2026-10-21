'''
Builds the two C libraries of gaiamockplus (loaded with ctypes) against GSL, which must be installed (e.g. brew install
gsl on macOS, apt install libgsl-dev on Debian/Ubuntu). The package metadata are in pyproject.toml.
'''
import shlex
import subprocess

from setuptools import Extension, setup

def gsl_flags():
    '''Compiler and linker flags for GSL, from gsl-config or pkg-config (falling back to the default search paths).'''
    for cflags_cmd, libs_cmd in ((['gsl-config', '--cflags'], ['gsl-config', '--libs']), (['pkg-config', '--cflags', 'gsl'], ['pkg-config', '--libs', 'gsl'])):
        try:
            cflags = subprocess.run(cflags_cmd, capture_output=True, text=True, check=True).stdout
            libs = subprocess.run(libs_cmd, capture_output=True, text=True, check=True).stdout
            return shlex.split(cflags), shlex.split(libs)
        except (OSError, subprocess.CalledProcessError):
            continue
    return [], ['-lgsl', '-lgslcblas']

cflags, libs = gsl_flags()
setup(ext_modules=[Extension('gaiamockplus.' + name, sources=['gaiamockplus/%s.c' % name], extra_compile_args=['-O2'] + cflags,
                             extra_link_args=libs + ['-lm']) for name in ('kepler_solve_astrometry', 'kepler_solve_planets')])
