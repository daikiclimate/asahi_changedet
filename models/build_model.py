from .featmodel import ImgModel, MultiHeadModel


def build_model(config):
    model_name = config.model
    model_type = config.type
    model_head = config.head
    input_channel = config.input_channel
    if model_name == "vgg":
        model = ImgModel(input_channel=input_channel)
    if model_name == "multi_head_vgg":
        model = MultiHeadModel(input_channel=input_channel)

    return model
