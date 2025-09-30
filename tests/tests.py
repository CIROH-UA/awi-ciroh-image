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
        self.assertIn('jupyter_server   : 2.13.0', result.stdout)
        self.assertIn('jupyterlab       :', result.stdout)
        pass

    pass

if __name__ == '__main__':
    #unittest.main(exit=False)
    #unittest.main(TestPyNGIAB(), exit=False)
    #unittest.main(TestNGIABDataPreprocess(), exit=False)

    ''' Test cases to make sure `ngen` and utilities are available in commandline '''
    ngiab_cmd_tests = unittest.TestLoader().discover('.', pattern = 'test_ngiab_cmd*.py')
    unittest.TextTestRunner(verbosity=1).run(ngiab_cmd_tests)

    ''' Test cases for PyNGIAB module '''
    pyngiab_tests = unittest.TestLoader().discover('.', pattern = 'test_pyngiab*.py')
    unittest.TextTestRunner(verbosity=1).run(pyngiab_tests)
    
    ''' Test cases for data preprocessing via ngiab_data_cli module '''
    ngiab_data_preprocess_tests = unittest.TestLoader().discover('.',
                                                                 pattern = 'test_ngiab_data*.py')
    unittest.TextTestRunner(verbosity=1).run(ngiab_data_preprocess_tests)

    ''' Test cases for TEEHR module (https://github.com/RTIInternational/teehr/)
    Note: TEEHR has its own test cases which are invoked from commandline
    Please see /tests/test-entrypoint.sh
    '''
    # teehr_tests = unittest.TestLoader().discover('.', pattern = 'test_pyngiab*.py')
    # unittest.TextTestRunner(verbosity=1).run(teehr_tests)
    
    pass
