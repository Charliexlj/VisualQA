from transformers import PreTrainedModel, PretrainedConfig
import torchxrayvision as xrv
from medclip import MedCLIPVisionModelViT
from medclip.dataset import MedCLIPFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torchvision
import numpy as np
import PIL
import PIL.Image as Image
from transformers.image_utils import is_torch_tensor
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import skimage
import torchvision.transforms as transforms

class MedCLIP_Preprocess(MedCLIPFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.size=224
        self.crop_size=(224,224)
        self.image_mean = .5862785803043838
        self.image_std = .27950088968644304
    # Resolve transformers conflict, copy from https://github.com/huggingface/transformers/blob/2b9e252b16396c926dad0e3c31802b4af8004e93/src/transformers/image_utils.py
    def _ensure_format_supported(self, image):
        if not isinstance(image, (PIL.Image.Image, np.ndarray)) and not is_torch_tensor(image):
            raise ValueError(
                f"Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and "
                "`torch.Tensor` are."
            )

    def to_pil_image(self, image, rescale=None):
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        self._ensure_format_supported(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim > 3:
                image = np.squeeze(image, axis=0)
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image.squeeze(2))
        return image

    def convert_rgb(self, image):
        """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
        self._ensure_format_supported(image)
        if not isinstance(image, PIL.Image.Image):
            return image

        return image.convert("RGB")

    def rescale(self, image: np.ndarray, scale: Union[float, int]) -> np.ndarray:
        """
        Rescale a numpy image by scale amount
        """
        self._ensure_format_supported(image)
        return image * scale

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
            channel_first (`bool`, *optional*, defaults to `True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_torch_tensor(image):
            image = image.numpy()

        rescale = isinstance(image.flat[0], np.integer) if rescale is None else rescale

        if rescale:
            image = self.rescale(image.astype(np.float32), 1 / 255.0)

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def expand_dims(self, image):
        """
        Expands 2-dimensional `image` to 3 dimensions.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to expand.
        """
        self._ensure_format_supported(image)

        # Do nothing if PIL image
        if isinstance(image, PIL.Image.Image):
            return image

        if is_torch_tensor(image):
            image = image.unsqueeze(0)
        else:
            image = np.expand_dims(image, axis=0)
        return image

    def normalize(self, image, mean, std, rescale=False):
        """
        Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
        if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to normalize.
            mean (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (`List[float]` or `np.ndarray` or `torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
            rescale (`bool`, *optional*, defaults to `False`):
                Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
                happen automatically.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image, rescale=True)
        # If the input image is a PIL image, it automatically gets rescaled. If it's another
        # type it may need rescaling.
        elif rescale:
            if isinstance(image, np.ndarray):
                image = self.rescale(image.astype(np.float32), 1 / 255.0)
            elif is_torch_tensor(image):
                image = self.rescale(image.float(), 1 / 255.0)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)
        # if image.ndim == 2:
        #     image = image[None]
        # print(mean)
        # print(std)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def resize(self, image, size, resample=None, default_to_square=True, max_size=None):
        """
        Resizes `image`. Enforces conversion of input to PIL.Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
                matched to this.

                If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
                `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
                this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                The filter to user for resampling.
            default_to_square (`bool`, *optional*, defaults to `True`):
                How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
                square (`size`,`size`). If set to `False`, will replicate
                [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
                with support for resizing only the smallest edge and providing an optional `max_size`.
            max_size (`int`, *optional*, defaults to `None`):
                The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
                greater than `max_size` after being resized according to `size`, then the image is resized again so
                that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
                edge may be shorter than `size`. Only used if `default_to_square` is `False`.

        Returns:
            image: A resized `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        if isinstance(size, list):
            size = tuple(size)

        if isinstance(size, int) or len(size) == 1:
            if default_to_square:
                size = (size, size) if isinstance(size, int) else (size[0], size[0])
            else:
                width, height = image.size
                # specified size only for the smallest edge
                short, long = (width, height) if width <= height else (height, width)
                requested_new_short = size if isinstance(size, int) else size[0]

                if short == requested_new_short:
                    return image

                new_short, new_long = requested_new_short, int(requested_new_short * long / short)

                if max_size is not None:
                    if max_size <= requested_new_short:
                        raise ValueError(
                            f"max_size = {max_size} must be strictly greater than the requested "
                            f"size for the smaller edge size = {size}"
                        )
                    if new_long > max_size:
                        new_short, new_long = int(max_size * new_short / new_long), max_size

                size = (new_short, new_long) if width <= height else (new_long, new_short)

        return image.resize(size, resample=resample)

    def center_crop(self, image, size):
        """
        Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
        size given, it will be padded (so the returned result has the size asked).

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n_channels, height, width) or (height, width, n_channels)):
                The image to resize.
            size (`int` or `Tuple[int, int]`):
                The size to which crop the image.

        Returns:
            new_image: A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n_channels,
            height, width).
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            size = (size, size)

        # PIL Image.size is (width, height) but NumPy array and torch Tensors have (height, width)
        if is_torch_tensor(image) or isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = self.expand_dims(image)
            image_shape = image.shape[1:] if image.shape[0] in [1, 3] else image.shape[:2]
        else:
            image_shape = (image.size[1], image.size[0])

        top = (image_shape[0] - size[0]) // 2
        bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        left = (image_shape[1] - size[1]) // 2
        right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.

        # For PIL Images we have a method to crop directly.
        if isinstance(image, PIL.Image.Image):
            return image.crop((left, top, right, bottom))

        # Check if image is in (n_channels, height, width) or (height, width, n_channels) format
        channel_first = True if image.shape[0] in [1, 3] else False

        # Transpose (height, width, n_channels) format images
        if not channel_first:
            if isinstance(image, np.ndarray):
                image = image.transpose(2, 0, 1)
            if is_torch_tensor(image):
                image = image.permute(2, 0, 1)

        # Check if cropped area is within image boundaries
        if top >= 0 and bottom <= image_shape[0] and left >= 0 and right <= image_shape[1]:
            return image[..., top:bottom, left:right]

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_shape = image.shape[:-2] + (max(size[0], image_shape[0]), max(size[1], image_shape[1]))
        if isinstance(image, np.ndarray):
            new_image = np.zeros_like(image, shape=new_shape)
        elif is_torch_tensor(image):
            new_image = image.new_zeros(new_shape)

        top_pad = (new_shape[-2] - image_shape[0]) // 2
        bottom_pad = top_pad + image_shape[0]
        left_pad = (new_shape[-1] - image_shape[1]) // 2
        right_pad = left_pad + image_shape[1]
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[
            ..., max(0, top) : min(new_image.shape[-2], bottom), max(0, left) : min(new_image.shape[-1], right)
        ]

        return new_image

    def flip_channel_order(self, image):
        """
        Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
        `image` to a NumPy array if it's a PIL Image.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
                be first.
        """
        self._ensure_format_supported(image)

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        return image[::-1, :, :]

    def rotate(self, image, angle, resample=None, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
        counter clockwise around its centre.

        Args:
            image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
                The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
                rotating.

        Returns:
            image: A rotated `PIL.Image.Image`.
        """
        resample = resample if resample is not None else PIL.Image.NEAREST

        self._ensure_format_supported(image)

        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.rotate(
            angle, resample=resample, expand=expand, center=center, translate=translate, fillcolor=fillcolor
        )

    def __call__(self, images):
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        new_images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = self.to_pil_image(image)
                new_images.append(image)
        images = new_images

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]

        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # add a RGB dim for each image
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        return images

    def pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        xrv_emb_size = 1024,
        lm_emb_size = 4096,
        clip_emb_size = 512,
        vocab_size = 32768,
        language_encoder_path = None,
        image_token_index = 12,
        IMG_START = 20,
        IMG_END = 21,
        CLS_START = 22,
        CLS_END = 23,
        IGNORE_INDEX = 100,
        clip_ckp = '/data/cl2920/VLM_Child_Models/',
        **kwargs,
    ):
        self.xrv_emb_size = xrv_emb_size
        self.lm_emb_size = lm_emb_size
        self.clip_emb_size = clip_emb_size
        self.vocab_size = vocab_size
        self.language_encoder_path = language_encoder_path
        self.image_token_index = image_token_index
        self.IMG_START = IMG_START,
        self.IMG_END = IMG_END,
        self.CLS_START = CLS_START,
        self.CLS_END = CLS_END,
        self.IGNORE_INDEX = IGNORE_INDEX
        self.clip_ckp = clip_ckp
        
        lm_config={
            "architectures": [
                "MistralForCausalLM"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "model_type": "mistral",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.42.0.dev0",
            "use_cache": True,
            "vocab_size": 32768
            }
        self.lm_config = PretrainedConfig.from_dict(lm_config)

        super().__init__(**kwargs)

class VisionEncoder(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.pathologies = self.model.pathologies

    def forward(self, x):
        prob = self.model(x)
        return prob

class MedCLIP(MedCLIPVisionModelViT):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(medclip_checkpoint=config.clip_ckp)
        self.lm_proj = nn.Linear(config.clip_emb_size, config.lm_emb_size)

    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values)
        img_embeds = output['pooler_output']
        if project:
            img_embeds = self.projection_head(img_embeds)
        img_embeds = self.lm_proj(img_embeds)
        return img_embeds

class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config, path=None):
        super().__init__(config)
        self.config = config

        self.vision_encoder = VisionEncoder(config)
        
        self.vision_ppv80 = [0.72, # Atelectasis
                             0.62, # Cardiomegaly
                             0.544, # Consolidation
                             0.6, # Edema
                             0.68, # Effusion
                             0.5, # Emphysema
                             0.54, # Enlarged Cardiomediastinum
                             0.60, # Fibrosis
                             0.85, # Fracture
                             0.6, # Hernia
                             0.6, # Infiltration
                             0.54, # Lung Lesion
                             0.6, # Lung Opacity
                             0.9, # Mass
                             0.6, # Nodule
                             0.6, # Pleural_Thickening
                             0.62, # Pneumonia
                             0.6] # Pneumothorax
        self.medclip = MedCLIP(config)

        self.language_tokenizer = AutoTokenizer.from_pretrained(...)
        self.language_tokenizer.pad_token = self.language_tokenizer.eos_token
        self.language_tokenizer.padding_side = 'right'

        self.language_encoder = AutoModelForCausalLM.from_pretrained(...)
        self.lm_head = self.language_encoder.lm_head

        self.resize = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        self.clip_preprocess = MedCLIP_Preprocess()

        self.use_classification = True
        self.file_names = None
        self.img_input = 'tensor'
        
        if path is not None:
            self.load_state_dict(torch.load(path))
            
        # transformers config bug
        if len(torch.tensor(self.config.IMG_START).shape) == 2:
            self.config.IMG_START = self.config.IMG_START[0]
            self.config.IMG_END = self.config.IMG_END[0]
            self.config.CLS_START = self.config.CLS_START[0]
            self.config.CLS_END = self.config.CLS_END[0]
            
    def get_input_embeddings(self):
        return self.language_encoder.get_input_embeddings()
        
    def eval(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def disable_classification(self):
        self.use_classification = False

    def train_proj(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.medclip.lm_proj.parameters():
            param.requires_grad = True

    def finetune(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.language_encoder.parameters():
            param.requires_grad = True
        for param in self.medclip.lm_proj.parameters():
            param.requires_grad = True

    def image_transform(self, x):
        device = x.device
        imgs = []
        for img in x:
            img = img.detach().cpu().numpy()
            if len(img.shape) == 3:
                if img.shape[0] in [1, 3]:
                    img = img.transpose(1, 2, 0)
                img = img.mean(2)
            img = xrv.datasets.normalize(img, 255)
            img = self.resize(img[None, ...])
            imgs.append(img)
        imgs = np.array(imgs)
        return torch.from_numpy(imgs).to(device)

    def encode_image_for_classification(self, x):
        if self.use_classification:
            if self.img_input == 'tensor':
                img = self.image_transform(x)
            else:
                img = x
                # self.save_image(img[0], 'test_densenet.jpg')
            probs = self.vision_encoder(img).detach().cpu().tolist()
            pathologies = [[str(self.vision_encoder.pathologies[i]) for i, (p, t) in enumerate(zip(prob, self.vision_ppv80)) if p > t] for prob in probs]
            pathologies = [' '.join(path)+' ' for path in pathologies]
            # print('\n', pathologies)
            pathologies = self.language_tokenizer.batch_encode_plus(
                    pathologies, return_tensors="pt",
                    add_special_tokens=False, padding=True
                    ).to(x.device)
            emb = self.language_encoder.model.embed_tokens(pathologies['input_ids'])
            mask = pathologies['attention_mask']
        else:
            emb = torch.empty((x.shape[0], 0, self.config.lm_emb_size)).to(x.device)
            mask = torch.empty((x.shape[0], 0), dtype=torch.int).to(x.device)
        return emb, mask

    def encode_image_for_language_alignment(self, x):
        if self.img_input == 'tensor':
            device = x.device
            all_imgs = []
            for img in x:
                img = img.detach().cpu().numpy()
                img = self.clip_preprocess(img[None])
                all_imgs.append(img[0])
            all_imgs = np.array(all_imgs)
            all_imgs = torch.from_numpy(all_imgs).to(device)
        else:
            all_imgs = x
            # self.save_image(all_imgs[0], 'test_medclip.jpg')
        all_imgs = self.medclip(all_imgs)
        return all_imgs[:,None,:]
    
    def save_image(self, data, file_path):
        # Convert PyTorch tensor to NumPy array if necessary
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.dtype == np.float32 or data.dtype == np.float64:
            if data.max() <= 1.0:
                data = (data * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        
        skimage.io.imsave(file_path, data)

    def generate_input_embeddings(self, input_ids, images, attention_mask=None, labels=None, past_key_values=None):
        '''
        Generate embeddings for a batch of input sequences containing text and optional images.

        Args:
        - input_ids (torch.Tensor): Tensor of input IDs with shape (batch_size, seq_len).
        - images (torch.Tensor): Tensor of images with shape (num_images, 3, 224, 224).
        - attention_mask (torch.Tensor): Tensor of attention masks with shape (batch_size, seq_len).
        - labels (torch.Tensor): Tensor of labels with shape (batch_size, seq_len).

        Returns:
        - torch.Tensor: A tensor containing the concatenated embeddings of text and images with
        adjustments for sequences with the image token, shape (batch_size, new_seq_len, embedding_dim).
        '''
        images_cls, images_align = None, None
        if images is not None:
            if images.shape[1]==1 and len(images.shape)==2:
                self.img_input = 'index'
                images_cls_tmp = []
                images_align_tmp = []
                for f_idx in images:
                    img = skimage.io.imread(self.file_names[f_idx[0]])
                    assert len(img.shape) == 2, f"Images in wrong shape, should be (w, h), got {img.shape}"
                    img_cls = torch.tensor(img).to(input_ids.device)
                    if len(img_cls.shape) == 2:
                        img_cls = img_cls.unsqueeze(0)
                    img_cls = self.image_transform(img_cls[None,...])
                    assert img_cls.shape == (1, 1, 224, 224), f"Images in wrong shape, should be (1, 1, 224, 224), got {img_cls.shape}"
                    images_cls_tmp.append(img_cls)
                    
                    device = images.device
                    img = self.clip_preprocess(img[None])
                    assert img[0].shape == (1, 224, 224), f"Images in wrong shape, should be (1, 224, 224), got {img.shape}"
                    images_align_tmp.append(img[0])
                images_cls = torch.cat(images_cls_tmp, dim=0)
                images_align_tmp = np.array(images_align_tmp)
                images_align = torch.from_numpy(images_align_tmp).to(device)
                assert len(images_cls.shape) == 4, f"Images in wrong shape, should be (batch_size, 1, w, h), got {img_cls.shape}"
                assert len(images_align.shape) == 4, f"Images in wrong shape, should be (batch_size, w, h), got {images_align.shape}"

        if past_key_values is not None:
            new_attention_mask = torch.full((attention_mask.shape[0], past_key_values[-1][-1].shape[-2]+1),
                                True,
                                dtype=attention_mask.dtype,
                                device=attention_mask.device)
            return input_ids, None, new_attention_mask, labels

        if images is not None:
            if self.img_input == 'index':
                img_emb_class, img_emb_mask = self.encode_image_for_classification(images_cls)
                img_emb_align = self.encode_image_for_language_alignment(images_align)
            else:
                img_emb_class, img_emb_mask = self.encode_image_for_classification(images)
                img_emb_align = self.encode_image_for_language_alignment(images)

            total_image_tokens = (input_ids == self.config.image_token_index).sum()
            assert total_image_tokens == len(images), f"Number of image tokens ({total_image_tokens}) must match the batch size of the images tensor ({len(images)})."

        num_images = torch.sum(input_ids==self.config.image_token_index, axis=1)

        assert len(input_ids.shape) == 2, f"Input_IDs in wrong shape, should be (batch_size, seq_len), got {input_ids.shape}"

        if attention_mask is None:
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int, device=input_ids.device)

        new_input_emb, new_labels = [], [] if labels is not None else None
        new_attention_mask = []
        img_idx = 0
        for batch_idx, input_id in enumerate(input_ids):
            # Check if there is an image token in the sequence
            img_token_positions = (input_id == self.config.image_token_index).nonzero().squeeze()
            if img_token_positions.numel() == 0:
                token_embeddings = self.language_encoder.model.embed_tokens(input_id)
                new_input_emb.append(token_embeddings)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_attention_mask.append(attention_mask[batch_idx])
                continue

            # Handle sequences with image token
            img_pos = img_token_positions.tolist()
            if isinstance(img_pos, int): img_pos = [img_pos]
            num_img = len(img_pos)

            input_emb, cur_labels = [], [] if labels is not None else None
            cur_attention_mask = []
            current_pos = 0
            for i, pos in enumerate(img_pos):
                cur_img_idx = img_idx+i
                input_emb.append(
                    self.language_encoder.model.embed_tokens(input_id[current_pos:pos]))
                input_emb.append(
                    self.language_encoder.model.embed_tokens(
                        torch.tensor(self.config.IMG_START).to(img_emb_class.device)))
                input_emb.append(img_emb_align[cur_img_idx])
                input_emb.append(
                    self.language_encoder.model.embed_tokens(
                        torch.tensor(self.config.IMG_END).to(img_emb_class.device)))
                input_emb.append(
                    self.language_encoder.model.embed_tokens(
                        torch.tensor(self.config.CLS_START).to(img_emb_class.device)))
                input_emb.append(img_emb_class[cur_img_idx])
                input_emb.append(
                    self.language_encoder.model.embed_tokens(
                        torch.tensor(self.config.CLS_END).to(img_emb_class.device)))

                if labels is not None:
                    cur_labels.append(labels[batch_idx, current_pos:pos])
                    cur_labels.append(torch.full((img_emb_align.shape[1]+img_emb_class.shape[1]+4,),
                                   self.config.IGNORE_INDEX,
                                   device=labels.device,
                                   dtype=labels.dtype))


                cur_attention_mask.append(attention_mask[batch_idx, current_pos:pos])
                cur_attention_mask.append(torch.full((4,), # IMG_START, img_emb_align, IMG_END, CLS_START
                                True,
                                device=attention_mask.device,
                                dtype=attention_mask.dtype))
                cur_attention_mask.append(img_emb_mask[cur_img_idx])
                cur_attention_mask.append(torch.full((1,), # CLS_END
                                True,
                                device=attention_mask.device,
                                dtype=attention_mask.dtype))

                current_pos = pos + 1

            img_idx += num_img

            input_emb.append(
                self.language_encoder.model.embed_tokens(input_id[current_pos:]))
            # for emb in input_emb:
            #     print(emb.shape)
            input_emb = torch.cat(input_emb, dim=0)
            new_input_emb.append(input_emb)

            if labels is not None:
                cur_labels.append(labels[batch_idx, current_pos:])
                cur_labels = torch.cat(cur_labels)
                new_labels.append(cur_labels)

            cur_attention_mask.append(attention_mask[batch_idx, current_pos:])
            cur_attention_mask = torch.cat(cur_attention_mask)
            new_attention_mask.append(cur_attention_mask)

        if torch.unique(num_images).numel() > 1:
            max_num = torch.max(num_images)
            for i, num in enumerate(num_images):
                pad_size = (max_num-num)*(img_emb_class.shape[1]+5)+num
                new_input_emb[i] = torch.cat([
                    new_input_emb[i],
                    self.language_encoder.model.embed_tokens(
                        torch.full((pad_size,),
                            self.language_tokenizer.pad_token_id,
                            dtype=input_ids[i].dtype,
                            device=input_ids[i].device)
                    )])
                if labels is not None:
                    new_labels[i] = torch.cat([
                        new_labels[i],
                        torch.full((pad_size,),
                            self.config.IGNORE_INDEX,
                            dtype=new_labels[i].dtype,
                            device=new_labels[i].device)
                    ])
                new_attention_mask[i] = torch.cat([
                    new_attention_mask[i],
                    torch.full((pad_size,),
                        False,
                        dtype=new_attention_mask[i].dtype,
                        device=new_attention_mask[i].device)
                ])

        new_input_emb = torch.stack(new_input_emb)
        if labels is not None:
            new_labels = torch.stack(new_labels)
        new_attention_mask = torch.stack(new_attention_mask)

        return None, new_input_emb, new_attention_mask, new_labels




    def forward(self,
            input_ids: torch.LongTensor = None,
            attention_mask = None,
            past_key_values = None,
            inputs_embeds = None,
            labels = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            images = None,
            return_dict = None
            ):
        input_ids, inputs_embeds, attention_mask, labels = self.generate_input_embeddings(
            input_ids=input_ids, images=images, attention_mask=attention_mask, labels=labels, past_key_values=past_key_values)

        outputs = self.language_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.IGNORE_INDEX)
            loss = loss_fct(shift_logits, shift_labels)
            
        # if return_dict:
        #     return loss, logits
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    def conversation(self, question, new_image=None, chat_dict=None, device='cuda'):
        INSTRUCT = '''[INST] {} [/INST]'''
        INSTRUCT_IMG = '''[INST] {} [control_10] [/INST]'''
        if chat_dict is None:
            chat_dict = self.apply_template(question, new_image)
        else:
            if chat_dict['output'] is not None:
                chat_dict['text'] += chat_dict['output']
            if new_image is not None:
                chat_dict['images'] = torch.cat([chat_dict['images'], new_image]).to(device)
                chat_dict['text'] += INSTRUCT_IMG.format(question)
            else:
                chat_dict['text'] += INSTRUCT.format(question)
        with torch.no_grad():
            self.language_tokenizer.truncation_side = 'left'
            input_ids = self.language_tokenizer.encode(chat_dict['text'], return_tensors='pt', max_length=512, truncation=True).to(device)
            if torch.sum(input_ids==self.config.image_token_index) != chat_dict['images'].shape[0]:
                chat_dict['images'] = chat_dict['images'][-torch.sum(input_ids==self.config.image_token_index):]
            # print(input_ids.shape, chat_dict['images'].shape)
            output = self.generate(input_ids, images=chat_dict['images'], max_length=512, eos_token_id=self.language_tokenizer.eos_token_id)
            output_text = self.language_tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        chat_dict['output'] = output_text
        return chat_dict
        
    def apply_template(self, question, image=None, device='cuda'):
        if image is None:
            TEMPLATE = '''[INST]You are a medical multimodel tasked with question answering and interpreting chest x-ray images.\n\n{}\n[/INST]'''
        else:
            TEMPLATE = '''[INST]You are a medical multimodel tasked with question answering and interpreting chest x-ray images.\n\n{}\n\n[control_10]\n[/INST]'''
        return {
            'text': TEMPLATE.format(question),
            'images': image.to(device) if image is not None else None,
            'output': None
        }
        