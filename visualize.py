import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
from prepare.processor import process_fn

color = np.array([
        [180, 119, 31], 
        [14, 127, 255], 
        [44, 160, 44], 
        [40, 39, 214], 
        [189, 103, 148], 
        [75, 86, 140], 
        [194, 119, 227], 
        [127, 127, 127], 
        [34, 189, 188], 
        [207, 190, 23]
    ]).astype(np.uint8)

gmm = GMM(n_components=1, random_state=42)

def draw_ellipse(img, position, covariance, color, datas, times=4, thick=3):
    x, y, nx, ny = datas    

    # Convert covariance to principal axes
    if len(covariance.shape) == 1 or covariance.shape[0] != covariance.shape[1]:
        covariance = np.diag(covariance)

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    width  = (width * (nx / (np.max(x) - np.min(x)))).astype(np.int32)
    height = (height * (ny / (np.max(y) - np.min(y)))).astype(np.int32)

    # Draw the Ellipse
    for nsig in range(1, times):
        cv2.ellipse(img, position, [nsig * width, nsig * height], angle, 0, 360, (255, 255, 255), thick)
        cv2.ellipse(img, position, [nsig * width, nsig * height], angle, 0, 360, color, thick - 1)

def queries_draw_gaussian(
        z,
        x, 
        y,
        nx,
        ny,
        dataloader,
        device,
        transform_q=None, 
        circle_size=3
    ):
    datas = {}
    for queries, docs in dataloader:
        q_x, d_x, c = process_fn(queries, docs, device)
        if transform_q != None:
            x_out = transform_q(q_x)
            x1 = x_out.mean[:, 0]
            x2 = x_out.mean[:, 1]
        else:
            x1 = q_x[:, 0]
            x2 = q_x[:, 1]

        for j in range(x1.shape[0]):
            _c  = c[j].item()
            _x1 = x1[j].item()
            _x2 = x2[j].item()
            datas[_c] = datas[_c] + [[_x1, _x2]] if _c in datas else [[_x1, _x2]]
    
    # print(datas)
    for c in datas:
        data = np.array(datas[c])
        if data.shape[0] < 2:
            continue
        gmm.fit(data)

        x1, x2 = gmm.means_[0]
        _x1 = int(((x1 - np.min(x)) / (np.max(x) - np.min(x))) * nx)
        _x2 = int(((x2 - np.min(y)) / (np.max(y) - np.min(y))) * ny)
        
        _color = [int(color[int(c)][i]) for i in range(3)]
        draw_ellipse(z, [_x1, _x2], gmm.covariances_[0], _color, [x, y, nx, ny], 2, 2)

def queries_draw(
        z,
        x, 
        y,
        nx,
        ny,
        dataloader,
        device,
        transform_q=None, 
        circle_size=3
    ):
    for queries, docs in dataloader:
        q_x, d_x, c = process_fn(queries, docs, device)
        if transform_q != None:
            x_out = transform_q(q_x)
            x1 = x_out.mean[:, 0]
            x2 = x_out.mean[:, 1]
        else:
            x1 = q_x[:, 0]
            x2 = q_x[:, 1]

        x1 = (((x1 - np.min(x)) / (np.max(x) - np.min(x))) * nx).int()
        x2 = (((x2 - np.min(y)) / (np.max(y) - np.min(y))) * ny).int()

        for j in range(x1.shape[0]):
            _x1 = x1[j].item()
            _x2 = x2[j].item()
            if _x1 < 0 or _x1 > nx or _x2 < 0 or _x2 > ny:
                ...
            else:
                cv2.circle(z, (_x1, _x2), circle_size, (255, 255, 255), -1)
                _color = [int(color[int(c[j].item())][i]) for i in range(3)]
                cv2.circle(z, (_x1, _x2), circle_size - 1, _color, -1)

def documents_draw(
        z,
        x, 
        y,
        nx,
        ny,
        dataloader,
        device,
        transform_d=None, 
        circle_size=8
    ):
    doc_dict = {}
    sigma    = None
    for queries, docs in dataloader:
        q_x, d_x, c = process_fn(queries, docs, device)
        if c in doc_dict:
            continue
        doc_dict[c] = True

        if transform_d != None:
            out_x = transform_d(d_x)
            sigma = out_x.sigma.cpu().detach().numpy()
            x1 = out_x.mean[:, 0]
            x2 = out_x.mean[:, 1]
        else:
            x1 = d_x[:, 0]
            x2 = d_x[:, 1]

        x1 = (((x1 - np.min(x)) / (np.max(x) - np.min(x))) * nx).int()
        x2 = (((x2 - np.min(y)) / (np.max(y) - np.min(y))) * ny).int()

        for j in range(x1.shape[0]):
            _x1 = x1[j].item()
            _x2 = x2[j].item()
            if _x1 < 0 or _x1 > nx or _x2 < 0 or _x2 > ny:
                ...
            else:
                cv2.circle(z, (_x1, _x2), circle_size, (255, 255, 255), -1)
                _color = [int(color[int(c[j].item())][i]) for i in range(3)]
                cv2.circle(z, (_x1, _x2), circle_size - 1, _color, -1)
                if sigma is not None:
                    _sig = sigma[j]
                    draw_ellipse(z, [_x1, _x2], _sig, _color, [x, y, nx, ny], 2)

def find_boundary(
        dataloader,
        device,
        transform_q=None, 
    ):

    min_x1, max_x1, min_x2, max_x2 = 1000000, -1000000, 1000000, -1000000
    for queries, docs in dataloader:
        q_x, d_x, c = process_fn(queries, docs, device)

        if transform_q != None:
            out_x = transform_q(q_x)
            x1 = out_x.mean[:, 0]
            x2 = out_x.mean[:, 1]
        else:
            x1 = q_x[:, 0]
            x2 = q_x[:, 1]
        
        if min_x1 >= torch.min(x1): min_x1 = torch.min(x1)
        if min_x2 >= torch.min(x2): min_x2 = torch.min(x2)

        if max_x1 < torch.max(x1): max_x1 = torch.max(x1)
        if min_x2 < torch.max(x2): max_x2 = torch.max(x2)

    mean_x1 = ((max_x1 - min_x1) / 2 + min_x1).item()
    mean_x2 = ((max_x2 - min_x2) / 2 + min_x2).item()
    max_x = max([(max_x1 - min_x1) / 2, (max_x2 - min_x2) / 2]).item() * 1.25

    return mean_x1 + max_x, mean_x2 + max_x, mean_x1 - max_x, mean_x2 - max_x

def predict_draw(
        x_max, 
        y_max, 
        samples, 
        model,
        dataloader,
        device,
        transform_q=None, 
        transform_d=None, 
        circle_size=3,
        zoom_auto=False
    ):
    nx, ny = (samples, samples)

    if zoom_auto:
        x_max, y_max, x_min, y_min = find_boundary(dataloader, device, transform_q)
    else:
        x_min  = x_max * -1
        y_min  = y_max * -1

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(x, y)

    z = np.zeros([nx, ny, 3])

    # if zoom_auto == False:
    with torch.no_grad():
        xv  = xv.reshape(1, -1)
        yv  = yv.reshape(1, -1)
        idx = np.array([xv, yv]).T.reshape(-1, 2)
        idx = torch.from_numpy(idx).float()
        idx = idx.to(device)

        out = model(idx)
        pred  = torch.argmax(out.logits, axis=1).cpu().numpy()
        z = color[pred].reshape(nx, ny, 3)
    # else:
        # z = z + 255

    # draw queries
    queries_draw(z, x, y, nx, ny, dataloader, device, transform_q)

    # documents
    documents_draw(z, x, y, nx, ny, dataloader, device, transform_d)

    # draw queries gaussian
    queries_draw_gaussian(z, x, y, nx, ny, dataloader, device, transform_q)
                
    z = cv2.flip(z, 0)
    return z