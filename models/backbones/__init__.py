from .mobilenetv1 import mobilenet_v1_025, mobilenet_v1_050, mobilenet_v1
from .mobilenetv2 import mobilenet_v2
from .resnet import resnet18, resnet34, resnet50


from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    else:
        raise ValueError()