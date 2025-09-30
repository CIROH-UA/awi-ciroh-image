import unittest

class TestNGIABDataPreprocess(unittest.TestCase):
    def test_pyngiab_datapreprocess(self):
        from pyngiab import PyNGIABDataPreprocess
        try:
            p = PyNGIABDataPreprocess('cat-7080') \
                .subset() \
                .generate_forcings('2022-01-01', '2022-01-28') \
                .generate_realization() \
                .run()
        except Exception as e:
            print(f'An error occurred: {str(e)}')
            p = False
        self.assertEqual(p, True)
        pass
    pass
