import sys
import unittest

class TestJupyter(unittest.TestCase):
    def test_jupyter_run(self):
        import subprocess
        result = subprocess.run('jupyter --version',
                                capture_output=True,
                                text=True,
                                shell=True,
                                check=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('jupyter_server', result.stdout)
        self.assertIn('jupyterlab', result.stdout)
        pass

    pass

if __name__ == '__main__':
    failed = False

    ''' Test cases to make sure `ngen` and utilities are available in commandline '''
    ngiab_cmd_tests = unittest.TestLoader().discover('.', pattern = 'test_ngiab_cmd*.py')
    result = unittest.TextTestRunner(verbosity=1).run(ngiab_cmd_tests)
    if not result.wasSuccessful():
        failed = True

    ''' Test cases for PyNGIAB module '''
    pyngiab_tests = unittest.TestLoader().discover('.', pattern = 'test_pyngiab*.py')
    result = unittest.TextTestRunner(verbosity=1).run(pyngiab_tests)
    if not result.wasSuccessful():
        failed = True

    ''' Test cases for data preprocessing via ngiab_data_cli module '''
    ngiab_data_preprocess_tests = unittest.TestLoader().discover('.',
                                                                 pattern = 'test_ngiab_data*.py')
    result = unittest.TextTestRunner(verbosity=1).run(ngiab_data_preprocess_tests)
    if not result.wasSuccessful():
        failed = True

    ''' Test cases for TEEHR module (https://github.com/RTIInternational/teehr/)
    Note: TEEHR has its own test cases which are invoked from commandline
    Please see /tests/test-entrypoint.sh
    '''
    # teehr_tests = unittest.TestLoader().discover('.', pattern = 'test_pyngiab*.py')
    # result = unittest.TextTestRunner(verbosity=1).run(teehr_tests)

    if failed:
        sys.exit(1)
