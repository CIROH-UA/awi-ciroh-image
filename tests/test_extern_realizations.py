'''
End-to-end smoke runs of the extern models that have a sample dataset:
  * LSTM  (Python BMI, trained weights under /ngen/ngen/extern/lstm)
  * SUMMA (Fortran BMI, /dmod/shared_libs/libsummabmi.so, needs SUNDIALS/OpenBLAS)
The datasets are run with ngen-serial from the dataset directory, the same way
NGIAB-CloudInfra's CI runs them. SAC-SMA, Snow17, dHBV2 and Rust LSTM have no
sample dataset and are covered by the linkage/import checks in test_extern_models.py.
'''
import glob
import json
import os
import subprocess
import tarfile
import tempfile
import time
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ngen_env = {**os.environ, 'PATH': '/ngen/.venv/bin:' + os.environ['PATH']}
# Rely on ld.so.conf.d (ngen.conf, sundials.conf) as a real user's session would
ngen_env.pop('LD_LIBRARY_PATH', None)


class ExternRealizationBase:
    ''' Subclasses set ARCHIVE (zip or tar.gz in tests/) and DATASET (top-level dir) '''
    ARCHIVE = None
    DATASET = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        archive = os.path.join(HERE, self.ARCHIVE)
        if archive.endswith('.zip'):
            with zipfile.ZipFile(archive) as z:
                z.extractall(self._tmp.name)
        else:
            with tarfile.open(archive) as t:
                t.extractall(self._tmp.name, filter='data')
        self.data = os.path.join(self._tmp.name, self.DATASET)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ngen_serial_run(self):
        gpkg = glob.glob(os.path.join(self.data, 'config', '*.gpkg'))
        self.assertEqual(len(gpkg), 1, gpkg)
        start = time.time()
        result = subprocess.run(
            ['/dmod/bin/ngen-serial', gpkg[0], 'all', gpkg[0], 'all',
             'config/realization.json'],
            cwd=self.data, env=ngen_env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f'stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}')
        with open(os.path.join(self.data, 'config', 'realization.json')) as f:
            output_root = os.path.join(self.data, json.load(f).get('output_root', '.'))
        outputs = [f for ext in ('csv', 'nc')
                   for f in glob.glob(os.path.join(output_root, f'*.{ext}'))
                   if os.path.getmtime(f) >= start and os.path.getsize(f) > 0]
        self.assertTrue(outputs, f'ngen run produced no output files in {output_root}')


class TestLSTMRealization(ExternRealizationBase, unittest.TestCase):
    ARCHIVE = 'AWI-10154200-lstm.zip'
    DATASET = 'AWI-10154200-lstm'


class TestSUMMARealization(ExternRealizationBase, unittest.TestCase):
    ARCHIVE = 'AWI_16_01073000_011.tar.gz'
    DATASET = 'AWI_16_01073000_011'


if __name__ == '__main__':
    unittest.main()
