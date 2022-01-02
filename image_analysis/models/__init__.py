from .mlp import VanilaMLP

def get_backbone(model_name, num_classes):
    if model_name == "vanila":
        model = VanilaMLP(num_classes=num_classes)

    return model