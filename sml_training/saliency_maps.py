import numpy as np
import scipy as sp
import torch
import nibabel
import re
import matplotlib.pyplot as plt
from sklearn.base import is_classifier, is_regressor
import matplotlib as mpl
import torch.nn.functional as F

def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None,
                overlay_cmap=None, overlay_vmin=None, overlay_vmax=None):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis

    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`).
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if overlay_cmap is None:
        alpha_to_red_cmap = np.zeros((256, 4))
        alpha_to_red_cmap[:, 0] = 0.8
        alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)  # cmap.N-20)  # alpha values
        alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)
        overlay_cmap = alpha_to_red_cmap

    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            # print(axis_label, 'plotting slice', i_slice)

            plt.sca(ax)
            plt.axis('off')
            plt.imshow(sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax,
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white',
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

            if overlay is not None:
                plt.imshow(sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap,
                           vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)


def get_relevance_per_area(area_masks, relevance_map, normalize=True, merge_hemisphere=False):
    relevances = dict()
    for area, area_mask in area_masks.items():
        relevances[area] = np.sum(relevance_map * area_mask)
    if normalize:
        normalizing_cst = np.array(list(relevances.values())).sum()
        if normalizing_cst > 0:
            for area in area_masks:
                relevances[area] /= normalizing_cst  # make all areas sum to 1
        else:
            print("Warning: relevance scores sum to 0", flush=True)
            for area in area_masks:
                relevances[area] = 1./len(area_masks)
    if merge_hemisphere:
        # Merge left and right areas.
        for area in area_masks:
            if re.match(r"\w*_L$", area):
                area_RL = re.match(r"(\w*)_L$", area)[1] # extract area name without "_L"
                relevances[area_RL] = relevances[area_RL+"_L"] + relevances[area_RL+"_R"]
                del(relevances[area_RL+"_L"], relevances[area_RL+"_R"])
    return sorted(relevances.items(), key=lambda b:b[1], reverse=True)


def area_occlusion(model, image_tensors, area_masks, sklearn=False, target_class=None,
                   apply_softmax=True, is_classif=True,
                   cuda=False, occlusion_value=0):
    """
    Perform brain area occlusion to determine the relevance of each image pixel
    for the classification/regression decision. Return a relevance heatmap over the input image.

    Args:
        model (torch.nn.Module or sklearn): The pytorch/sklearn model. If Pytorch, should be set to eval mode.
        image_tensors (torch.Tensor or numpy.ndarray): The images to run through the `model`
                                                      (channels first for Pytorch!).
        sklearn (boolean): whether the model to test is a scikit-learn model or not
        target_class (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        is_classif (boolean); Whether the model is a classifier or regressor
        cuda (boolean): Whether to run the computation on a cuda device.
        occlusion_value (float): constant value that will be set when an area is occluded

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """
    if sklearn:
        assert (is_classifier(model) and is_classif) or is_regressor(model), "Unknown sklearn model"
        image_tensors = np.array(image_tensors)
        if is_classif:
            if hasattr(model, "predict_proba"):
                outputs = model.predict_proba(image_tensors)
            # elif hasattr(model, "decision_function"):
            #     outputs = model.decision_function(image_tensors)
            else:
                raise ValueError("sklearn model %s has neither <predict_proba> nor <decision_fn>"%str(model.__class__))
        else:
            outputs = model.predict(image_tensors)
    else:
        image_tensors = torch.Tensor(image_tensors)  # convert numpy or list to tensor
        if cuda:
            image_tensors = image_tensors.cuda()
        outputs = model(image_tensors).squeeze().detach()

        if is_classif and apply_softmax:
            if len(outputs.shape) < 2:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)

    if is_classif:
        if len(outputs.shape) < 2: # output = p(x=1) for a binary classifier
            outputs = torch.stack([1-outputs, outputs], dim=1).detach()
        if sklearn:
            outputs_class = outputs.argmax(1)
        else:
            outputs_class = outputs.max(1)[1].data.cpu().numpy()
        if target_class is None:
            target_class = outputs_class
        unoccluded_prob = [outputs[i,j] for (i,j) in enumerate(target_class)]

        unoccluded_prob = np.stack(unoccluded_prob) if sklearn else \
            torch.stack(unoccluded_prob).to('cuda' if cuda else 'cpu')
    else:
        unoccluded_prob = outputs.squeeze()

    if sklearn:
        relevance_maps = np.zeros_like(image_tensors)
        area_masks = np.stack([m for m in area_masks.values()]).astype(np.bool)
    else:
        relevance_maps = torch.zeros_like(image_tensors[:, 0, :], requires_grad=False)
        if cuda:
            relevance_maps = relevance_maps.cuda()
        area_masks = torch.BoolTensor(np.stack([m for m in area_masks.values()]))
        if cuda:
            area_masks = area_masks.cuda()

    for area_mask in area_masks:
        image_tensors_occluded = image_tensors.copy() if sklearn else image_tensors.clone()
        if sklearn:
            image_tensors_occluded[:, area_mask] = occlusion_value
            outputs = model.predict_proba(image_tensors_occluded) if is_classif else \
                model.predict(image_tensors_occluded)
        else: # channel first
            image_tensors_occluded[:, :, area_mask] = occlusion_value
            outputs = model(image_tensors_occluded).squeeze().detach()
            if is_classif and apply_softmax:
                if len(outputs.shape)>=2:
                    outputs = torch.softmax(outputs, dim=1)
                else:
                    outputs = torch.sigmoid(outputs)
            if is_classif and len(outputs.shape)<2:
                outputs = torch.stack([1-outputs, outputs], dim=1)

        if is_classif:
            occluded_prob = [outputs[i,j] for (i,j) in enumerate(target_class)]
            occluded_prob = np.stack(occluded_prob) if sklearn else \
                torch.stack(occluded_prob).to('cuda' if cuda else 'cpu')
        else:
            occluded_prob = outputs

        for i in range(len(image_tensors)):
            if sklearn:
                relevance_maps[i, area_mask] = (unoccluded_prob - occluded_prob)[i]
            else:
                relevance_maps[i, area_mask] = (unoccluded_prob - occluded_prob.detach())[i]

    if any([torch.max(torch.abs(r)) < 1e-6 for r in relevance_maps]):
        raise ValueError()

    if not sklearn:
        relevance_maps = relevance_maps.detach().cpu().numpy()
    if is_classif:
        relevance_maps = np.maximum(relevance_maps, 0)
    else:
        relevance_maps = np.abs(relevance_maps)
    return relevance_maps


def sensitivity_analysis(model, image_tensors, postprocess='abs', is_classif=True, sklearn=False,
                         apply_softmax=True, cuda=False):
    """
    Perform sensitivity analysis (via backpropagation; Simonyan et al. 2014) to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Args:
        model (torch.nn.Module or sklearn): The pytorch/sklearn model. Should be set to eval mode if Pytorch.
                                            If sklearn model, should be a linear one so the gradient is directly given
                                            by the weights.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used
                                                 in Simonyan et al. 2014, `'square'` is used in Montavon et al. 2018.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        appl (None or 'binary' or 'categorical'): Whether the output format of the `model` is binary
                                                         (i.e. one output neuron with sigmoid activation) or categorical
                                                         (i.e. multiple output neurons with softmax activation).
                                                         If `None` (default), infer from the shape of the output.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """
    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square' (got %s)"%postprocess)

    if sklearn:
        assert is_classifier(model) or is_regressor(model), "Unknown sklearn model"
        assert 'sklearn.linear_model' in model.__module__, "sklearn model must be linear"
        relevance_map = model.coef_
    else:
        # Forward pass.
        image_tensors = torch.tensor(image_tensors, requires_grad=True, device=("cuda" if cuda else "cpu"))
        outputs = model(image_tensors).squeeze()
        if is_classif:
            if apply_softmax:
                outputs = torch.sigmoid(outputs) if len(outputs.shape) < 2 else torch.softmax(outputs, dim=1)
            if len(outputs.shape) < 2:  # output = p(x=1) for a binary classifier
                outputs = torch.stack([1 - outputs, outputs], dim=1)
            output_class = outputs.max(1)[1]
        else:
            output_class = outputs

        # Backward pass.
        model.zero_grad()
        if is_classif:
            one_hot_output = F.one_hot(output_class, num_classes=outputs.shape[1])
            if cuda:
                one_hot_output = one_hot_output.cuda()
            outputs.backward(gradient=one_hot_output)
        else:
            outputs.backward(gradient=torch.ones_like(outputs))

        relevance_map = image_tensors.grad.data.cpu().numpy().squeeze()

    if any([np.max(np.abs(r)) < 1e-6 for r in relevance_map]):
        raise ValueError()

    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map ** 2
    elif postprocess is None:
        return relevance_map


def parse_atlas_mapping(path_to_mapping):
    # From a .txt file, parse each line and get a dict(index: str) indicating the region
    atlas_map = dict()
    with open(path_to_mapping, "r") as f:
        all_lines = f.readlines()
        atlas_map = {int(l.split(" ")[0]): l.split(" ")[1] for l in all_lines}
    return atlas_map

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def get_brain_area_masks(data_size, path_to_atlas, path_to_mapping_atlas, transforms=None):
    brain_map = nibabel.load(path_to_atlas).get_fdata()
    brain_areas = np.unique(brain_map)[1:]  # omit background
    mapping_atlas = parse_atlas_mapping(path_to_mapping_atlas)
    brain_areas_masked = dict()
    for area in brain_areas:
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = resize_image(area_mask, data_size, interpolation=0).astype(np.bool)
        if transforms is not None:
            area_mask = transforms(area_mask).astype(np.bool)
        brain_areas_masked[mapping_atlas[area]] = area_mask
    return brain_areas_masked

