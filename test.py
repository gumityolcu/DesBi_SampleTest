import argparse
import os
from pathlib import Path
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

from dualxda import DualDA
from representer import RepresenterPoints


from torch.nn import Module, Sequential, Conv2d, ReLU, Linear, Flatten, LeakyReLU, MaxPool2d
import os
import torch

"""

Basic Model:
Only linear
Conv+linear
With and without biases

VGG with and without biases

"""


class BasicModel(Module):
    empty = {
        'num': 0,
        'padding': 0,
        'kernel': 0,
        'stride': 0,
        'features': 0
    }

    def __init__(self, input_shape, num_classes, convs=None, fc=None, bias=True, leaky=False):
        super(BasicModel, self).__init__()
        if convs is None:
            convs = BasicModel.empty
        if fc is None:
            fc = BasicModel.empty
        x = torch.zeros(size=input_shape)

        if isinstance(convs['kernel'], int):
            convs['kernel'] = [convs['kernel'] for _ in range(convs['num'])]
        if isinstance(convs['padding'], int):
            convs['padding'] = [convs['padding'] for _ in range(convs['num'])]
        if isinstance(convs['stride'], int):
            convs['stride'] = [convs['stride'] for _ in range(convs['num'])]
        if isinstance(convs['features'], int):
            convs['features'] = [convs['features'] for _ in range(convs['num'])]
        assert convs['num'] == len(convs['kernel'])
        assert convs['num'] == len(convs['padding'])
        assert convs['num'] == len(convs['stride'])
        assert convs['num'] == len(convs['features'])
        assert fc['num'] == len(fc['features'])
        activation_class = LeakyReLU if leaky else ReLU
        self.convs = convs
        self.fc = fc
        self.bias = bias
        self.leaky = leaky
        self.features = Sequential()
        for c in range(convs['num']):
            module = Conv2d(x.shape[0], convs['features'][c], kernel_size=convs['kernel'][c],
                            padding=convs['padding'][c], stride=convs['stride'][c], bias=bias)
            with torch.no_grad():
                x = module(x)
            self.features.add_module(name=f'conv-{c}',
                                     module=module)
            self.features.add_module(name=f"relu-{c}", module=activation_class())
        self.features.add_module(name='flatten', module=Flatten())
        x = torch.flatten(x)
        last_features = x.shape[0]
        for i in range(fc['num']):
            self.features.add_module(name=f'fc-{i}',
                                     module=Linear(in_features=last_features, out_features=fc['features'][i],
                                                   bias=bias))
            last_features = fc['features'][i]
            self.features.add_module(name=f"relu-{convs['num'] + i}", module=activation_class())
        self.classifier = Linear(in_features=last_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class BasicConvModel(BasicModel):
    default_convs = {
        'num': 3,
        'padding': 0,
        'kernel': 3,
        'stride': 1,
        'features': [5, 10, 5]
    }
    default_fc = {
        'num': 2,
        'features': [500, 100]
    }

    def __init__(self, input_shape, num_classes, convs=None, fc=None, leaky=False):
        if convs is None:
            convs = BasicConvModel.default_convs
        if fc is None:
            fc = BasicConvModel.default_fc

        super(BasicConvModel, self).__init__(
            num_classes=num_classes,
            convs=convs,
            fc=fc,
            leaky=leaky,
            input_shape=input_shape
        )

def load_model():
    params={
                'convs': {
                        'num': 3,
                        'padding': 0,
                        'kernel': 3,
                        'stride': 1,
                        'features': [5, 10, 5]
                    },

                'fc' : {
                    'num': 2,
                    'features': [500, 100]
                },

                'input_shape':(1,28,28)
            }

    return BasicConvModel(num_classes=10, **params)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

inverse_mnist_transform = transforms.Compose([transforms.Normalize(mean=(0.,),
                                                                std=(1 / 0.3081,)),
                                        transforms.Normalize(mean=(-0.1307,),
                                                                std=(1.,)),
                                        ])

def make_figures(method_name, explainer, test_sample, attr, target, class_names):
    out_dir = Path("figures") / method_name
    explainer.da_figure(
        test_sample=test_sample,
        inv_transform=inverse_mnist_transform,
        class_names=class_names,
        attr=attr,
        fname=f"{method_name}_da",
        nsamples=5,
        save_path=str(out_dir),
    )
    explainer.xda(
        test_sample=test_sample,
        inv_transform=inverse_mnist_transform,
        class_names=class_names,
        attr=attr,
        attr_target=target,
        fname=f"{method_name}_xda",
        nsamples=5,
        composite="EpsilonPlusFlat",
        canonizer=None,
        save_path=str(out_dir),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_index=10
    transform = mnist_transform
    train = datasets.MNIST(
        root=".",
        train=True,
        transform=transform,
        download=True,
    )
    test_set = datasets.MNIST(
        root=".",
        train=False,
        transform=transform,
        download=True,
    )

    model = load_model()
    model.load_state_dict(torch.load("mnist_checkpoint.pt", map_location=device)["model_state"])
    test_sample, test_label = test_set[test_index]
    test_batch = test_sample.unsqueeze(0).to(device)
    target = torch.tensor([test_label], device=device)
    class_names = [str(i) for i in range(10)]

    dualda = DualDA(
        model=model,
        dataset=train,
        classifier_layer="classifier",
        device=str(device),
        cache_dir="./dualda_cache",
    )
    dual_attr = dualda.attribute(test_batch, target)
    print(f"DualDA attribution shape: {tuple(dual_attr.shape)}")
    print(f"DualDA coefficients shape: {tuple(dualda.coefficients.shape)}")

    representer = RepresenterPoints(
        model=model,
        dataset=train,
        classifier_layer="classifier",
        device=str(device),
        cache_dir="./representer_cache",
        max_iter=1000,
    )
    representer_attr = representer.attribute(test_batch, target)
    print(f"Representer attribution shape: {tuple(representer_attr.shape)}")
    print(f"Representer coefficients shape: {tuple(representer.coefficients.shape)}")

    make_figures(
        method_name="dualda",
        explainer=dualda,
        test_sample=test_sample,
        attr=dual_attr[0],
        target=test_label,
        class_names=class_names,
    )
    make_figures(
        method_name="representer",
        explainer=representer,
        test_sample=test_sample,
        attr=representer_attr[0],
        target=test_label,
        class_names=class_names,
    )
    print(f"Saved figures under {Path('figures').resolve()}")


if __name__ == "__main__":
    main()
