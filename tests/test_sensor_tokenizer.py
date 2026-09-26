import numpy as np

from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord
from safegrip.universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig


def _sine(rate, freq=2.0, duration=1.0, quantity="acceleration"):
    t = np.arange(0.0, duration, 1.0 / rate)
    x = np.sin(2 * np.pi * freq * t)
    unit = "m/s^2" if quantity == "acceleration" else "strain"
    return SensorChannel(x, t, SensorMeta(quantity, unit, "lateral", "vehicle_body", rate))


def test_different_sampling_rates_produce_compatible_feature_dimension():
    tok = UniversalSensorTokenizer(TokenizerConfig(patch_duration_sec=0.5))
    dims = []
    counts = []
    for rate in (20.0, 100.0, 1000.0):
        r = SensorRecord([_sine(rate)], sequence_id=str(rate))
        z = tok.tokenize(r)
        dims.append(z.features.shape[1])
        counts.append(z.num_tokens)
    assert dims == [tok.feature_dim] * 3
    assert counts == [2, 2, 2]


def test_multiple_channels_are_tokenized_without_fixed_schema():
    tok = UniversalSensorTokenizer()
    r = SensorRecord([
        _sine(20.0),
        _sine(100.0, quantity="strain"),
    ])
    z = tok.tokenize(r)
    assert len(np.unique(z.channel_ids)) == 2
    assert z.num_tokens >= 2


def test_numeric_context_becomes_token():
    from safegrip.universal.schema import ContextValue
    r = SensorRecord([], context=[ContextValue(2.0, SensorMeta("pressure", "bar", "scalar", "tire"))])
    z = UniversalSensorTokenizer().tokenize(r)
    assert z.num_tokens == 1
    assert z.channel_ids[0] == -2
