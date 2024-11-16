from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as TF


def _get_cl_tfm():
    sz = 512
    sz_tfm = TF.Compose([TF.Resize(sz), TF.CenterCrop(sz)])
    gs_tfm = TF.Grayscale(num_output_channels=3)
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    norm_tfm = TF.Normalize(mean, std)
    # tfm = TF.Compose([tfm, TF.ToTensor(), get_IN_norm_tfm()])
    return TF.Compose([sz_tfm, gs_tfm, TF.ToTensor(), norm_tfm])

class SludgeAnalyzer:
    def __init__(self):
        self.cl_model = torch.load('classification.pt', weights_only=False)
        self.od_model = YOLO("object_detection.pt")
        self.classnames = np.array(['Annelida', 'Ciliophora', 'Nematoda', 'Rotifera', 'Sarcodina'])
        self.cl_tfm = _get_cl_tfm()
        self.od_tfm = TF.Grayscale(num_output_channels=3)

    def predict(self, image:Image.Image):
        assert not self.cl_model.training
        inp = self.cl_tfm(image).unsqueeze(0)
        with torch.no_grad():
            out = self.cl_model(inp)[0]
        cl_pred = self.classnames[out > 0].tolist()
        inp = self.od_tfm(image)
        r = self.od_model.predict(inp)[0]
        return {'classes': cl_pred, 'image': Image.fromarray(r.plot()[:,:,::-1])}