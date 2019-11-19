"""Unit tests for world 3 model."""
from whynot.simulators.world3.simulator import *


def get_world3_context():
    """Load a javascript context for world3"""
    ctx = PyMiniRacerContext()
    ctx.eval(WORLD3_JS_CODE)
    return ctx


def test_set_state():
    """Check setting initial_state."""
    ctx = get_world3_context()
    initial_state = State(land_fertility=3434)

    set_state(ctx, initial_state)
    assert ctx.eval(f"landFertility.initVal") == 3434

    # Run the model and check if land fertility is correctly used.
    ctx.eval("fastRun()")
    states, _ = decode_states(ctx)
    assert states[0].land_fertility == 3434


def test_set_config():
    """Check setting config + intervention works as expected."""
    ctx = get_world3_context()
    config = Config(service_capital_output_ratio=23)
    intervention = Intervention(nonrenewable_resource_usage_factor=2)

    set_config(ctx, config, intervention)

    for param in dataclasses.asdict(config):
        if param in ["policy_year", "start_time", "end_time", "delta_t"]:
            continue

        if param == "service_capital_output_ratio":
            assert ctx.eval("serviceCapitalOutputRatio.before") == 23
            assert ctx.eval("serviceCapitalOutputRatio.after") == 23
        elif param == "nonrenewable_resource_usage_factor":
            assert (
                ctx.eval("nonrenewableResourceUsageFactor.before")
                == config.nonrenewable_resource_usage_factor
            )
            assert ctx.eval("nonrenewableResourceUsageFactor.after") == 2
        else:
            assert ctx.eval(f"{to_camel_case(param)}.before") == ctx.eval(
                f"{to_camel_case(param)}.after"
            )


def test_setup():
    """Test setup of world3 simulator."""
    for idx in range(3):
        ctx = get_world3_context()
        initial_state = State(land_fertility=idx * 1111)
        config = Config(service_capital_output_ratio=idx * 22)
        intervention = Intervention(nonrenewable_resource_usage_factor=idx * 22)

        set_state(ctx, initial_state)
        set_config(ctx, config, intervention)

        assert ctx.eval(f"landFertility.initVal") == idx * 1111
        assert ctx.eval("serviceCapitalOutputRatio.before") == idx * 22
        assert ctx.eval("serviceCapitalOutputRatio.after") == idx * 22
        assert (
            ctx.eval("nonrenewableResourceUsageFactor.before")
            == config.nonrenewable_resource_usage_factor
        )
        assert ctx.eval("nonrenewableResourceUsageFactor.after") == idx * 22
