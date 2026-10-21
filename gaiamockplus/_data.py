'''
Locations of the compiled C libraries and of the data files, and the one-time download of the Gaia scanning-law data
(healpix_scans/, the scan times and angles at 49152 sky positions; see the gaiamock README).

The data are kept in the folder data_dir(): the environment variable GAIAMOCKPLUS_DATA if it is set (e.g. a folder
that already contains a healpix_scans/ download), and otherwise ~/.gaiamockplus.
'''
import glob
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

SCAN_DATA_URL = 'https://caltech.box.com/shared/static/4f7q6qdh0bku881bzvzxc4cm5u0902cf.zip' # gaiamock's healpix_scans.zip (Caltech Box)
SCAN_DATA_SHA256 = '86eff185a2e1e135157b69703d285216bacd8a50119e5a1f0e8ccf940ff1dc21'
SCAN_DATA_BYTES = 286007203
N_SCAN_FILES = 49152 # healpix_64_0.fits ... healpix_64_49151.fits

def compiled_library(name):
    '''
    Path of a compiled C library of the package: name.so (compiled by hand) or name.<platform tag>.so (built by pip).
    '''
    paths = [os.path.join(PACKAGE_DIR, name + '.so')] + sorted(glob.glob(os.path.join(PACKAGE_DIR, name + '.*.so')))
    for path in paths:
        if os.path.exists(path):
            return path
    raise ValueError('The compiled library %s was not found in %s. Install gaiamockplus with pip (pip install gaiamockplus, or '
                     'pip install -e . in the repository), which compiles it.' % (name, PACKAGE_DIR))

def data_dir():
    '''The folder holding healpix_scans/ (and any optional data files): $GAIAMOCKPLUS_DATA, or ~/.gaiamockplus.'''
    return os.path.abspath(os.path.expanduser(os.environ.get('GAIAMOCKPLUS_DATA') or os.path.join('~', '.gaiamockplus')))

def scan_data_available():
    '''Whether the scanning-law data are in data_dir()/healpix_scans/.'''
    scans = os.path.join(data_dir(), 'healpix_scans')
    return all(os.path.isfile(os.path.join(scans, 'healpix_64_%d.fits' % k)) for k in (0, N_SCAN_FILES - 1))

def scan_data_dir():
    '''The folder with the scanning-law data, downloading them first (once) if they are not there.'''
    if not scan_data_available():
        download_scan_data()
    return os.path.join(data_dir(), 'healpix_scans')

def download_scan_data(force=False):
    '''
    Download the Gaia scanning-law data (healpix_scans.zip, 286 MB; 984 MB unzipped) from Caltech Box into
    data_dir()/healpix_scans/. This is only needed once: later sessions use the saved copy. It is done automatically
    the first time the data are needed. force=True downloads them again.
    '''
    import fcntl
    target_dir = data_dir()
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, '.download.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX) # one download at a time (e.g. from parallel processes)
        if scan_data_available() and not force:
            return
        print('gaiamockplus: downloading the Gaia scanning-law data (healpix_scans: %.0f MB download, 984 MB unzipped) to %s.\n'
              'This is a one-time download; later sessions will use the saved copy. To keep the data elsewhere (e.g. an existing '
              'healpix_scans/ folder), set the environment variable GAIAMOCKPLUS_DATA to the folder that contains healpix_scans/.'
              % (SCAN_DATA_BYTES/1e6, target_dir), flush=True)
        tmp = tempfile.mkdtemp(dir=target_dir, prefix='.download_')
        try:
            zip_path = os.path.join(tmp, 'healpix_scans.zip')
            sha = hashlib.sha256()
            with urllib.request.urlopen(SCAN_DATA_URL) as response, open(zip_path, 'wb') as out:
                total = int(response.headers.get('Content-Length') or SCAN_DATA_BYTES)
                done, shown = 0, -1
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    sha.update(chunk)
                    done += len(chunk)
                    percent = int(100*done/total)
                    if percent != shown and percent % 5 == 0:
                        print('\r  downloaded %3d%% (%.0f of %.0f MB)' % (percent, done/1e6, total/1e6), end='', flush=True)
                        shown = percent
            print(flush=True)
            if sha.hexdigest() != SCAN_DATA_SHA256:
                raise RuntimeError('The downloaded scanning-law data do not match the expected checksum (the download may be '
                                   'incomplete, or the file at %s may have changed). Nothing was saved; please try again.' % SCAN_DATA_URL)
            print('  unzipping...', flush=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
            n = len(glob.glob(os.path.join(tmp, 'healpix_scans', 'healpix_64_*.fits')))
            if n != N_SCAN_FILES:
                raise RuntimeError('healpix_scans.zip contains %d files instead of %d.' % (n, N_SCAN_FILES))
            final = os.path.join(target_dir, 'healpix_scans')
            if os.path.exists(final):
                shutil.rmtree(final)
            os.replace(os.path.join(tmp, 'healpix_scans'), final)
            print('gaiamockplus: the scanning-law data are saved in %s.' % final, flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
