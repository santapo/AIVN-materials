from .mlp import VanilaMLP
from .mixer import mixer_s16

def get_backbone(model_name, num_classes):
    if model_name == "vanila":
        model = VanilaMLP(num_classes=num_classes)
    if model_name == "mixer":
        model = mixer_s16(num_classes=num_classes)
    return model