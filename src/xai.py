import os
import copy
import numpy as np
import matplotlib.cm as mpl_color_map
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# References:
# - CAM: http://cnnlocalization.csail.mit.edu/
# - GitHub CAM: https://github.com/zhoubolei/CAM

# - Grad-CAM: https://arxiv.org/pdf/1512.04150.pdf
# - GitHub Grad-CAM: https://github.com/utkuozbulak/pytorch-cnn-visualizations



def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    elif isinstance(im, torch.Tensor):
        im = T.functional.to_pil_image(im)
    im.save(path)


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, output_path, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(output_path, file_name + '.png')
    save_image(gradient, path_to_file)


def apply_colormap_on_image(org_img, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_img.size()[-2:])
    org_img = T.functional.to_pil_image(org_img) if isinstance(org_img, torch.Tensor) else org_img
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_img.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_class_activation_images(org_img, activation_map, filepath):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        filepath (str): File path of the exported image
    """
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    save_image(org_img, filepath + '_sample_image.png')
    save_image(heatmap, filepath + '_heatmap.png')
    save_image(heatmap_on_image, filepath + '_heatmap_on_image.png')
    save_image(activation_map, filepath + '_activation_grayscale.png')


class VanillaBackprop():
    # From https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_img, target_class):
        # Forward
        model_output = self.model(input_img)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


class ClassActivationMapping_ORG:
    # From https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    def __init__(self, model, target_layer=None):
        self.model = model
        self.features_blobs = []
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers(target_layer)
    
    def hook_layers(self, target_layer):
        def hook_function(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())

        # Register hook to the first layer
        self.model.conv_net.register_forward_hook(hook_function)
        # self.model._modules.get(target_layer).register_forward_hook(hook_function)
        # last_layer = list(self.model.features._modules.items())[-1][1]   # Originally [0][1]
        # last_layer.register_forward_hook(hook_function)
    
    def return_cam(self, feature_conv, weight_softmax, class_idx, input_img, upsample: bool = False):
        # generate the class activation maps upsample to 256x256
        if upsample:
            size_upsample = (256, 256)
        else:
            size_upsample = (input_img.shape[2], input_img.shape[3])
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            # output_cam.append(cv2.resize(cam_img, size_upsample))
            cam_img = np.uint8(Image.fromarray(cam_img).resize((size_upsample[0], size_upsample[1]), Image.LANCZOS))/255
            output_cam.append(cam_img)
        return output_cam
    
    def generate_cam(self, input_img, device=None, target_class=None):
        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())
        input_img.requires_grad = True
        logit = self.model(input_img)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        cams = self.return_cam(self.features_blobs[0], weight_softmax, [idx[0]], input_img)
        # img = cv2.imread('test.jpg')
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(cams[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite('CAM.jpg', result)
        return cams


class ClassActivationMapping:
    # From https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.target_layer = target_layer


    def save_gradient(self, grad):
        self.gradients = grad


    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        # for module_pos, module in self.model._modules.items():
        #     x = module(x)
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer

        # for i, (module_name, module) in enumerate(self.model.named_modules()):
        #     print(f'Iteration {i}')
        #     print(module_name)
        #     x = module(x)
        #     print(f'Shape of x after module forward: {x.shape}')
        #     # if module_name == self.target_layer:
        #     if self.target_layer in module_name:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x

        x = self.model.conv_net(x)
        x.register_hook(self.save_gradient)
        conv_output = x
        return conv_output, x


    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


    def generate_cam(self, input_img, device, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.forward_pass(input_img)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = torch.zeros(1, model_output.size()[-1], dtype=torch.float, device=device)
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_img.shape[2],
                       input_img.shape[3]), Image.LANCZOS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_img[0].shape[1:])/np.array(cam.shape))
        return cam


def explain_vanilla_gradients(model, loader, out_path):
    backprop_grads = VanillaBackprop(model)
    batch = next(iter(loader))
    img, label, idx = batch
    grads = backprop_grads.generate_gradients(img, label)
    
    # Save colored and grayscale gradients
    save_gradient_images(grads, out_path, 'backprop_grads_color')
    grayscale_grads = convert_to_grayscale(grads)
    save_gradient_images(grayscale_grads, out_path, 'backprop_grads_grayscale')


def explain_cams(model, loader, org_loader, out_path, device, batch_step: int = 4, hook: str = 'hook'):
    cam = ClassActivationMapping(model, target_layer=hook)

    batch = next(iter(loader))
    org_batch = next(iter(org_loader))
    imgs, labels, idx = batch
    imgs = imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    len_imgs = len(imgs)

    for i in np.arange(0, len_imgs, batch_step):
        img = imgs[i].unsqueeze(0)
        label = labels[i].item()
        cams = cam.generate_cam(img, target_class=label, device=device)
        org_img = org_batch[0][i]
        save_class_activation_images(org_img, cams, out_path + f'/gradcam_{i+1}')