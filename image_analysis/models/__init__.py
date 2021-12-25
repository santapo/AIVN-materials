from .lenet import LeNet


def get_backbone(model_name, num_classes):
    if model_name == "lenet":
        model = LeNet(num_classes=num_classes)

    return model