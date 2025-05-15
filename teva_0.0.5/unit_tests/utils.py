import typing
import numpy as np
import pytest

def try_assert(message: str, assertion: bool):
    try:
        assert assertion
    except Exception as e:
        print(f"{message}: FAILED")
        raise e
    print(f"{message}: SUCCESS")
#
# def try_raise(function: typing.Callable,
#               ex: type(Exception) = None):
#     if ex is not None:
#         return pytest.raises(ex)
#     else:
#         # generic exception
#         return pytest.raises(Exception)

def expect_failure(message: str,
                   function: typing.Callable,
                   function_args: list = None,
                   function_kwargs: dict = None,
                   expected_exception: type(Exception) = None):
    if expected_exception is None:
        expected_exception = Exception

    try:
        with pytest.raises(expected_exception):
            if function_args is None and function_kwargs is None:
                function()
            elif function_args is None:
                function(**function_kwargs)
            elif function_kwargs is None:
                function(*function_args)
            else:
                function(*function_args, **function_kwargs)
    except Exception as e:
        print(f"{message}: FAILED")
        raise e
    print(f"{message}: SUCCESS")

def dict_match(dict1, dict2, threshold: float = 0):
    for key in dict1.keys():
        if key in dict2:
            if type(dict1[key]) is str and dict1[key] != dict2[key]:
                print(f"Exp: {dict1[key]}")
                print(f"Got: {dict2[key]}")
                return False
            if type(dict1[key]) is dict:
                return dict_match(dict1[key], dict2[key], threshold)
            else:
                return near(dict1[key], dict2[key], threshold)
    return False

def near(input1, input2, threshold: float = 0):
    diff = np.abs(input2 - input1)
    if isinstance(input1, np.ndarray):
        max_val = np.nanmax(diff)
    else:
        max_val = diff

    is_near = max_val <= threshold

    if not is_near:
        print(f"Input 1: {input1}")
        print(f"Input 2: {input2}")
        print(f"Nearness Diff: {diff}")
        print(f"Max Diff: {max_val}")

    return is_near
