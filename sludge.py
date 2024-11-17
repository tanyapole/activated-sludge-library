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

def _result_to_row(od_result):
    boxes_cls = od_result.boxes.cls.int().cpu().numpy()
    num_classes = 5
    row = torch.zeros(num_classes, dtype=int)
    for i in range(num_classes):
        row[i] = (boxes_cls == i).sum()
    return row

def _mean(od_result, count_result):
    count_result = count_result.relu().int()
    od_result = _result_to_row(od_result)
    mean = (count_result + od_result) / 2 + 1e-6
    mean = mean.round().int()
    return mean

def _pprint_count(od_result, count_result):
    count = _mean(od_result, count_result)
    classnames = ['Annelida', 'Ciliophora', 'Nematoda', 'Rotifera', 'Sarcodina']
    return {c: count[i].item() for i,c in enumerate(classnames)}

class SludgeAnalyzer:
    def __init__(self):
        self.cl_model = torch.load('classification.pt', weights_only=False)
        self.od_model = YOLO("object_detection.pt").to(torch.device('cpu'))
        self.counting_model = model = torch.load('counting.pt', weights_only=False)
        self.classnames = np.array(['Annelida', 'Ciliophora', 'Nematoda', 'Rotifera', 'Sarcodina'])
        self.cl_tfm = _get_cl_tfm()
        self.od_tfm = TF.Grayscale(num_output_channels=3)
        self.counting_tfm = self.cl_tfm

    def predict(self, image:Image.Image):
        assert not self.cl_model.training
        inp = self.cl_tfm(image).unsqueeze(0)
        with torch.no_grad():
            out = self.cl_model(inp)[0]
            count_pred = self.counting_model(inp)[0]
        cl_pred = self.classnames[out > 0].tolist()
        inp = self.od_tfm(image)
        r = self.od_model.predict(inp)[0]
        return {'classes': cl_pred, 
            'image': Image.fromarray(r.plot()[:,:,::-1]),
            'count': _pprint_count(r, count_pred)}