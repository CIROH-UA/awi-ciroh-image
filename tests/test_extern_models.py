'''
Static checks for the extern models bundled in the image (SUMMA, SAC-SMA, Snow17,
Rust LSTM, Python LSTM, dHBV2). These verify the libraries/packages/weights are
installed and loadable. The models that have a sample dataset (LSTM, SUMMA) are
additionally run end-to-end in test_extern_realizations.py.
'''
import glob
import os
import subprocess
import unittest

SHARED_LIBS = '/dmod/shared_libs'
VENV_PYTHON = '/ngen/.venv/bin/python'
EXTERN = '/ngen/ngen/extern'
ngen_env = {**os.environ, 'PATH': '/ngen/.venv/bin:' + os.environ['PATH']}
# Rely on ld.so.conf.d (ngen.conf, sundials.conf) as a real user's session would
ngen_env.pop('LD_LIBRARY_PATH', None)

BMI_LIBS = {
    'SUMMA': 'libsummabmi.so*',
    'SAC-SMA': 'libsacbmi.so*',
    'Snow17': 'libsnow17bmi.so*',
    'Rust LSTM': 'librust_lstm_1025.so',
}


def _find(pattern):
    return sorted(glob.glob(os.path.join(SHARED_LIBS, pattern)))


class TestExternLibraries(unittest.TestCase):
    def test_libraries_installed(self):
        for model, pattern in BMI_LIBS.items():
            with self.subTest(model=model):
                self.assertTrue(_find(pattern), f'{model}: no {pattern} in {SHARED_LIBS}')

    def test_libraries_fully_linked(self):
        ''' No unresolved shared-object dependencies (e.g. OpenBLAS, SUNDIALS, NetCDF-Fortran) '''
        for model, pattern in BMI_LIBS.items():
            for lib in _find(pattern):
                with self.subTest(model=model, lib=lib):
                    out = subprocess.run(['ldd', lib], capture_output=True, text=True,
                                         env=ngen_env, check=True).stdout
                    self.assertNotIn('not found', out, out)

    def _dlopen(self, lib, symbol=None):
        ''' dlopen in a clean subprocess (no LD_LIBRARY_PATH, venv interpreter) '''
        code = ('import ctypes, sys; l = ctypes.CDLL(sys.argv[1]); '
                'sys.exit(0 if len(sys.argv) < 3 or hasattr(l, sys.argv[2]) else 3)')
        cmd = [VENV_PYTHON, '-c', code, lib] + ([symbol] if symbol else [])
        return subprocess.run(cmd, capture_output=True, text=True, env=ngen_env)

    def test_libraries_loadable(self):
        for model, pattern in BMI_LIBS.items():
            for lib in _find(pattern)[:1]:
                with self.subTest(model=model, lib=lib):
                    result = self._dlopen(lib)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_bmi_symbols_exported(self):
        ''' Fortran/C BMI libraries are loaded by ngen via these entry points '''
        for model, symbol in (('SUMMA', 'register_bmi'), ('SAC-SMA', 'register_bmi'),
                              ('Snow17', 'register_bmi')):
            for lib in _find(BMI_LIBS[model])[:1]:
                with self.subTest(model=model):
                    result = self._dlopen(lib, symbol)
                    self.assertEqual(result.returncode, 0,
                                     f'{lib} does not export {symbol}: {result.stderr}')


class TestExternPythonModels(unittest.TestCase):
    def _venv_run(self, code):
        return subprocess.run([VENV_PYTHON, '-c', code], capture_output=True, text=True,
                              env=ngen_env)

    def test_python_models_importable(self):
        for module in ('lstm.bmi_lstm', 'dhbv2.bmi', 'dhbv2.mts_bmi', 'hydrodl2', 'dmg'):
            with self.subTest(module=module):
                result = self._venv_run(f'import {module}')
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_lstm_weights_installed(self):
        configs = glob.glob(f'{EXTERN}/lstm/trained_neuralhydrology_models/*/config.yml')
        self.assertTrue(configs, 'no LSTM trained-model config.yml files found')

    def test_dhbv2_weights_installed(self):
        base = f'{EXTERN}/dhbv2/ngen_resources/data'
        for model in ('dhbv_2', 'dhbv_2_mts'):
            with self.subTest(model=model):
                files = [f for f in glob.glob(f'{base}/{model}/model/{model}/**', recursive=True)
                         if os.path.isfile(f)]
                self.assertTrue(files, f'no weights extracted for {model}')
                self.assertTrue(all(os.path.getsize(f) > 0 for f in files))


if __name__ == '__main__':
    unittest.main()
