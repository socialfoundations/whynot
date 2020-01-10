"""Test suite for whynot framework."""
import numpy as np
import pytest

import whynot as wn
from whynot.framework import parameter


def test_parameter():
    """Test the parameter decorator."""
    # Ccase when no standard arguments
    @parameter(name="a", default=1, values=[1, 2], description="test a")
    @parameter(name="b", default=3, values=[3, 4], description="test b")
    def foo(a, b):
        return a + b

    def check_params(listed, returned):
        assert set(listed) == set(p.name for p in returned)

    check_params(["a", "b"], wn.framework.extract_params(foo, standard_args=[]))

    @parameter(name="b", default=3, values=[3, 4], description="test b")
    def bar(a, b):
        return a + b

    # Can omit a as a parameter if a is one of the "standard args"
    check_params(["b"], wn.framework.extract_params(bar, standard_args=["a"]))

    # If there are no standard args, raise a value error since "a" is not
    # listed as a parameter
    with pytest.raises(ValueError):
        wn.framework.extract_params(bar, standard_args=[])

    # If a parameter is added, but omitted form the signature, we should raise
    with pytest.raises(ValueError):

        @parameter(name="a", default=0)
        def baz(propensity):
            return propensity

    # If a parameter is also part of standard args, we should raise
    with pytest.raises(ValueError):

        @parameter(name="rng", default=0)
        def foobar(rng):
            return 2

        wn.framework.extract_params(foobar, standard_args=["rng"])

    # If values isn't specified, we should handle things sensible
    @parameter(name="a", default=1)
    @parameter(name="b", default=3, values=[3, 4])
    @parameter(name="c", default=4, values=np.arange(0.05, 0.95, 0.1))
    def values_test(a, b, c):
        return a + b + c

    param_collection = wn.framework.extract_params(values_test, standard_args=[])

    sampled = param_collection.sample()
    assert sampled["a"] == 1
    print(sampled)


def test_parameter_collection():
    """Test the parameter collection object."""
    from whynot.framework import ExperimentParameter

    p1 = ExperimentParameter(name="p1", default=1)
    collection = wn.framework.ParameterCollection(params=[p1])
    collection.add_parameter(ExperimentParameter(name="p2", default=2))

    projected = collection.project({"p1": 10})
    assert projected["p1"] == 10
    assert projected["p2"] == 2

    p3 = ExperimentParameter(name="p3", default=3)
    p4 = ExperimentParameter(name="p4", default=4)
    collection2 = wn.framework.ParameterCollection(params=[p3, p4])

    collection += collection2

    assert collection["p3"].default == 3

    # Ensure names are unique
    a = ExperimentParameter(name="a", default=3)
    aa = ExperimentParameter(name="a", default=4)
    with pytest.raises(ValueError):
        wn.framework.ParameterCollection(params=[a, aa])

    # Ensure names are unique after adding parameter
    with pytest.raises(ValueError):
        coll = wn.framework.ParameterCollection(params=[a])
        coll.add_parameter(aa)

    # Ensure names are unique if you add two collections
    with pytest.raises(ValueError):
        coll_one = wn.framework.ParameterCollection(params=[a])
        coll_two = wn.framework.ParameterCollection(params=[aa])
        new = coll_one + coll_two
