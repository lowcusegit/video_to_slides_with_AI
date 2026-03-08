import copy
import torch
from torchvision import models, transforms

# one may load a pretrained convnext model from the series available from torchvision
# pretrained_model=models.convnext_tiny(weights='IMAGENET1K_V1',progress=True).eval()

# then reuse with the feature output
class ConvNeXt_w_feature(torch.nn.Module):
    def __init__(
        self,
        _ConvNeXt_:models.ConvNeXt
    ) -> None:
        super().__init__()
        self.features = copy.deepcopy(_ConvNeXt_.features)
        self.avgpool = copy.deepcopy(_ConvNeXt_.avgpool)
        self.classifier = copy.deepcopy(_ConvNeXt_.classifier)
        
    def _forward_feature_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        # x = self.classifier(x)
        return x

    def forward_w_class(self, x: torch.Tensor):
        y = self._forward_feature_impl(x)
        x = self.classifier(y)
        return y,x
        
    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_feature_impl(x).flatten(start_dim=1)

# default inferance transform for ConvNeXt
ConvNeXt_model_input_size=224
ConvNeXt_transform= transforms.Compose([
    transforms.Resize(ConvNeXt_model_input_size, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])