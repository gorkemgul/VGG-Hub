import torch
import torch.nn as nn
import yaml

class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10, chosen_structure = "VGG16"):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_structure = self._read_from_yaml(config_path = "cfg/config.yaml", chosen_structure = chosen_structure)
        self.layers = self._make_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


    def _read_from_yaml(self, config_path, chosen_structure):
        with open(config_path, 'r') as cfg:
            model_config = yaml.safe_load(cfg)
            model_structure = model_config[chosen_structure]
            print(f"Chosen Model Structure:\n {chosen_structure} = {model_structure}")

        return model_structure


    def _make_layers(self):
        layers = []
        in_channels = self.in_channels

        for m in self.model_structure:
            if isinstance(m, int):
                out_channels = m
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
                        nn.ReLU()
                    )
                )
                in_channels = m

            elif isinstance(m, str):
                layers.append(
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                )

        return nn.Sequential(*layers)

    def _create_fcs(self):
        fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_classes)
        )
        return fcs


if __name__ == "__main__":
    model = VGG(chosen_structure = "VGG16")
    print(model)
    BATCH_SIZE = 3
    x = torch.randn(3, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([BATCH_SIZE, 1000])


