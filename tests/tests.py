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

test_data = '/tests/cat-7080'

class TestNGIAB(unittest.TestCase):
    def test_ngiab_install(self):
        import subprocess
        result = subprocess.run('/dmod/bin/ngen',
                                capture_output=True,
                                text=True,
                                check=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('NGen Framework', result.stdout)
        self.assertIn('NetCDF', result.stdout)
        self.assertIn('Fortran', result.stdout)
        self.assertIn('Python', result.stdout)
        pass

    def test_ngiab_serial(self):
        import subprocess
        result = subprocess.run('/dmod/bin/ngen-serial',
                                capture_output=True,
                                text=True,
                                check=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('NGen Framework', result.stdout)
        self.assertIn('NetCDF', result.stdout)
        self.assertIn('Fortran', result.stdout)
        self.assertIn('Python', result.stdout)
        self.assertIn('ngen-serial', result.stdout)
        pass

    def test_ngiab_parallel(self):
        import subprocess
        result = subprocess.run('/dmod/bin/ngen-parallel',
                                capture_output=True,
                                text=True,
                                check=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('NGen Framework', result.stdout)
        self.assertIn('NetCDF', result.stdout)
        self.assertIn('Fortran', result.stdout)
        self.assertIn('Python', result.stdout)
        self.assertIn('Parallel build', result.stdout)
        pass

    def test_ngiab_partition_generator(self):
        import subprocess
        result = subprocess.run('/dmod/bin/partitionGenerator',
                                capture_output=True,
                                text=True
                                #check=True
                                )
        self.assertEqual(result.returncode, 255)
        #self.assertIn('features from layer divides using ID column', result.stdout)
        self.assertIn('/dmod/bin/partitionGenerator <catchment_data_path> <nexus_data_path> <partition_output_name> <number of partitions> <catchment_subset_ids> <nexus_subset_ids>', result.stdout)
        pass

    def test_ngiab_run(self):
        #self.assertEqual(add(-1, -2), -3)
        pass


class TestPyNGIAB(unittest.TestCase):
    def test_pyngiab_serial(self):
        # import sys
        # sys.path.append('/ngen/pyngiab')
        from pyngiab import PyNGIAB

        data_dir = test_data
        try:
            test_ngiab = PyNGIAB(data_dir, serial_execution_mode=True)
            run = test_ngiab.run()
        except:
            run = False
        self.assertEqual(run, True)
        pass

    def test_pyngiab_parallel(self):
        # import sys
        # sys.path.append('/ngen/pyngiab')
        from pyngiab import PyNGIAB

        data_dir = test_data
        try:
            test_ngiab = PyNGIAB(data_dir)
            run = test_ngiab.run()
        except:
            run = False
        self.assertEqual(run, True)
        pass

class TestTeehr(unittest.TestCase):
    pass

class TestNGIABDataPreprocess(unittest.TestCase):
    def test_pyngiab_datapreprocess(self):
        # import sys
        # sys.path.append('/shared/pyngiab')
        from pyngiab import PyNGIABDataPreprocess
        try:
            p = PyNGIABDataPreprocess('cat-7080') \
                .subset() \
                .generate_forcings('2022-01-01', '2022-01-28') \
                .generate_realization() \
                .run()
        except:
            p = False
        self.assertEqual(p, True)
        pass
    pass

if __name__ == '__main__':
    ''' extract sample dataset '''
    import zipfile
    with zipfile.ZipFile(f'{test_data}.zip', 'r') as zip_ref:
        zip_ref.extractall('/tests/')
        pass

    unittest.main(exit=False)
    #unittest.main(TestPyNGIAB(), exit=False)
    #unittest.main(TestNGIABDataPreprocess(), exit=False)

    ''' Cleanup '''
    from pathlib import Path
    import shutil

    dirpath = Path(test_data)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
