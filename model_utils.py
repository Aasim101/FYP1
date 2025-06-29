import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import efficientnet_b3, vgg16, vgg19
import numpy as np

# Configuration
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = 'binary_models'
BASE_CHECKPOINTS = {
    efficientnet_b3: f'{BASE_DIR}/best_efficientnetb3.pth',
    vgg16:           f'{BASE_DIR}/best_vgg16.pth',
    vgg19:           f'{BASE_DIR}/best_vgg19.pth',
}
META_CHECKPOINT = f'{BASE_DIR}/stacking_meta.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Labels
CLASS_NAMES = ['Glioma', 'Meningioma', 'No_tumor', 'Pituitary']

# Ensemble model
def make_ensemble(base_models):
    class StackingEnsemble(nn.Module):
        def __init__(self, bases, hidden=64):
            super().__init__()
            self.bases = nn.ModuleList(bases)
            in_feats = len(bases) * NUM_CLASSES
            self.meta = nn.Sequential(
                nn.Linear(in_feats, hidden), nn.ReLU(), nn.Linear(hidden, NUM_CLASSES)
            )
        def forward(self, x):
            logits = [m(x) for m in self.bases]
            return self.meta(torch.cat(logits, dim=1))
    return StackingEnsemble(base_models).to(DEVICE)

@st.cache_resource
def load_models():
    # Load base models
    base_models = []
    for ctor, path in BASE_CHECKPOINTS.items():
        m = ctor(pretrained=False)
        if 'vgg' in ctor.__name__.lower(): in_f = m.classifier[6].in_features; m.classifier[6] = nn.Linear(in_f, NUM_CLASSES)
        else: in_f = m.classifier[1].in_features; m.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.to(DEVICE).eval()
        base_models.append(m)
    # Ensemble
    ensemble = make_ensemble(base_models)
    ensemble.load_state_dict(torch.load(META_CHECKPOINT, map_location=DEVICE))
    ensemble.eval()
    return ensemble


def preprocess_image(image):
    return transform(image).unsqueeze(0)


def predict(model, input_tensor):
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad(): out = model(input_tensor); probs = F.softmax(out, dim=1)[0].cpu().numpy(); idx = probs.argmax()
    return CLASS_NAMES[idx], {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}


def generate_gradcam(model, input_tensor, class_idx):
    def find_last_conv(net):
        last=None
        for m in net.modules():
            if isinstance(m, nn.Conv2d): last = m
        return last
    cams=[]
    for base in model.bases:
        layer = find_last_conv(base)
        if layer is None: continue
        activ=None; grads=None
        def f_hook(_, inp, out): nonlocal activ; activ = out
        def b_hook(_, gin, gout): nonlocal grads; grads = gout[0]
        layer.register_forward_hook(f_hook); layer.register_backward_hook(b_hook)
        base.zero_grad(); out = base(input_tensor.to(DEVICE)); (out[0,class_idx]).backward()
        w=grads.mean(dim=(2,3),keepdim=True)
        cam=F.relu((w*activ).sum(dim=1,keepdim=True))
        cam=F.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0].cpu().detach().numpy()
        cam-=cam.min(); cam/=cam.max() if cam.max()>0 else 1
        cams.append(cam)
        layer._forward_hooks.clear(); layer._backward_hooks.clear()
    avg=np.mean(cams,axis=0) if cams else np.ones((224,224))
    return avg
