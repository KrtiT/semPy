import numpy as np

import semopy
from semopy.effects import EffectMA, EffectStatic


def test_example_model():
    ex = semopy.examples.example_article
    desc, data = ex.get_model(), ex.get_data()
    model = semopy.Model(description=desc)

    # the preparations below are normally performed in `Model.fit`
    model.load(data=data)
    model.prepare_params()
    model.prepare_fiml()

    # a sanity check for `Model.obj_fiml`, `Model.grad_fiml`
    assert type(model.obj_fiml(model.param_vals)) == np.float64
    assert model.grad_fiml(model.param_vals).shape == model.param_vals.shape


def test_example_generalized_effects():
    ex = semopy.examples.example_article
    desc = ex.get_model()
    data, (k1, k2) = ex.get_data(random_effects=2, moving_average=True)

    ef = (EffectStatic("group", k1), EffectStatic("group", k2), EffectMA("time", 2))
    model = semopy.ModelGeneralizedEffects(description=desc, effects=ef)

    # TODO prepare model
    model.obj_fiml(model.param_vals)
