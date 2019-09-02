from squeezenet.networks.squeezenet import Squeezenet_Imagenet
from squeezenet.networks.squeezenet import Squeezenet_CIFAR

catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})


register(Squeezenet_Imagenet)
register(Squeezenet_CIFAR)
