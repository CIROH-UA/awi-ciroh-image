import os
import unittest
# local imports
from pyngiab import PyNGIAB

class TestPyNGIAB(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._test_data = '/tests/cat-7080'

        ''' extract sample dataset '''
        import zipfile
        with zipfile.ZipFile(f'{self._test_data}.zip', 'r') as zip_ref:
            zip_ref.extractall('/tests/')
            pass

        pass

    def test_pyngiab_serial(self):
        try:
            test_ngiab = PyNGIAB(self._test_data, serial_execution_mode=True)
            run = test_ngiab.run()
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            run = False
        self.assertEqual(run, True)
        pass

    def test_pyngiab_parallel(self):
        try:
            test_ngiab = PyNGIAB(self._test_data)
            run = test_ngiab.run()
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            run = False
        self.assertEqual(run, True)
        pass

    def __del__(self):
        ''' Cleanup '''
        from pathlib import Path
        import shutil

        dirpath = Path(self._test_data)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
