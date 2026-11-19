The expert algorithm is LaCAM, as described above. Example usage of its wrapper:
''
from lacam.inference import LacamInference, LacamInferenceConfig
class ExpertWrapper:
    def __init__(self, base_obj, withMaxSteps=False):
        self.base_obj = base_obj
        self.withMaxSteps = withMaxSteps

    def reset_states(self, env):
        self.base_obj.reset_states()

    def __getattr__(self, name):
        return getattr(self.base_obj, name)


def wrapped_class(cls, withMaxSteps=False):
    def _get_wrapped_class(config):
        return ExpertWrapper(cls(config), withMaxSteps=withMaxSteps)

    return _get_wrapped_class

inference_config = LacamInferenceConfig()
expert_algorithm = wrapped_class(LacamInference)
expert = expert_algorithm(inference_config)

actions = expert.act(observations)
''