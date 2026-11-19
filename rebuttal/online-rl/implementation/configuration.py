import bootstrap
import yaml

SPEC = yaml.safe_load((bootstrap.WORK/'configs/costtracer_128_proposed.yaml').read_text())
