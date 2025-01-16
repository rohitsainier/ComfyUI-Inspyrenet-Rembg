from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def create_gradient_background(size, color1, color2, direction="horizontal"):
    """Create a gradient background."""
    width, height = size
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)

    if direction == "horizontal":
        for x in range(width):
            r = int(color1[0] + (float(x)/(width-1))*(color2[0]-color1[0]))
            g = int(color1[1] + (float(x)/(width-1))*(color2[1]-color1[1]))
            b = int(color1[2] + (float(x)/(width-1))*(color2[2]-color1[2]))
            draw.line([(x, 0), (x, height)], fill=(r, g, b, 255))
    else:  # vertical
        for y in range(height):
            r = int(color1[0] + (float(y)/(height-1))*(color2[0]-color1[0]))
            g = int(color1[1] + (float(y)/(height-1))*(color2[1]-color1[1]))
            b = int(color1[2] + (float(y)/(height-1))*(color2[2]-color1[2]))
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    return image


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def add_text_overlay(image, text_config, font_path=None):
    """Add text overlay to an image using the text configuration."""
    draw = ImageDraw.Draw(image)

    try:
        if font_path:
            font = ImageFont.truetype(font_path, text_config["font_size"])
        else:
            font = ImageFont.truetype(
                text_config["font_family"], text_config["font_size"])
    except:
        font = ImageFont.load_default(text_config["font_size"])

    # Convert color if it's a hex string
    color = hex_to_rgb(text_config["text_color"])

    # Apply opacity
    color = (*color, int(255 * text_config["opacity"]))

    # Create a new image for rotated text if needed
    if text_config["rotation"] != 0:
        txt_img = Image.new('RGBA', image.size, (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text(text_config["position"], text_config["text_content"],
                      font=font, fill=color)
        rotated_txt = txt_img.rotate(text_config["rotation"], expand=True)
        image.paste(rotated_txt, (0, 0), rotated_txt)
    else:
        draw.text(text_config["position"], text_config["text_content"],
                  font=font, fill=color)

    return image


class TextOverlayConfig:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_content": ("STRING", {"default": "edit"}),
                "font_family": (["Inter", "Arial", "Times New Roman", "Helvetica", "Roboto"],),
                "text_color": ("STRING", {"default": "#FFFFFF"}),  # Hex color
                "text_placement": (["foreground", "background", "both"],),
                "x_position": ("FLOAT", {
                    "default": -12,
                    "min": -1000,
                    "max": 1000,
                    "step": 1
                }),
                "y_position": ("FLOAT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "step": 1
                }),
                "font_size": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 800,
                    "step": 1
                }),
                "font_weight": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 900,
                    "step": 100
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0,
                    "min": -360,
                    "max": 360,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("TEXT_CONFIG",)
    FUNCTION = "create_text_config"
    CATEGORY = "image/text"

    def create_text_config(self, text_content, font_family, text_color, text_placement,
                           x_position, y_position, font_size, font_weight, opacity, rotation):
        text_config = {
            "text_content": text_content,
            "font_family": font_family,
            "text_color": text_color,
            "text_placement": text_placement,
            "position": (x_position, y_position),
            "font_size": font_size,
            "font_weight": font_weight,
            "opacity": opacity,
            "rotation": rotation
        }
        return (text_config,)


class TextOverlayBatch:
    def __init__(self):
        self.text_configs = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # First required text config
                "text_config_1": ("TEXT_CONFIG",),
            },
            "optional": {
                "text_config_2": ("TEXT_CONFIG",),
                "text_config_3": ("TEXT_CONFIG",),
                "text_config_4": ("TEXT_CONFIG",),
                "text_config_5": ("TEXT_CONFIG",),
            }
        }

    RETURN_TYPES = ("TEXT_CONFIG_BATCH",)
    FUNCTION = "batch_text_configs"
    CATEGORY = "image/text"

    def batch_text_configs(self, text_config_1, text_config_2=None, text_config_3=None,
                           text_config_4=None, text_config_5=None):
        """
        Batches multiple text configurations into a single batch.

        Args:
            text_config_1: First text configuration (required)
            text_config_2: Second text configuration (optional)
            text_config_3: Third text configuration (optional)
            text_config_4: Fourth text configuration (optional)
            text_config_5: Fifth text configuration (optional)

        Returns:
            A tuple containing the batch of text configurations.
        """
        # Clear the existing batch
        self.text_configs.clear()

        # Add the required text config
        self.text_configs.append(text_config_1)

        # Add optional text configs if they exist
        for config in [text_config_2, text_config_3, text_config_4, text_config_5]:
            if config is not None:
                self.text_configs.append(config)

        # Return the batch
        return (self.text_configs,)


class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],),
                "output_type": (["original", "foreground", "background", "replace_background"],),
                "background_mode": (["color", "gradient", "image"],),
            },
            "optional": {
                "background_color": ("STRING", {"default": "#000000"}),
                "gradient_color1": ("STRING", {"default": "#000000"}),
                "gradient_color2": ("STRING", {"default": "#FFFFFF"}),
                "gradient_direction": (["horizontal", "vertical"],),
                "background_image": ("IMAGE",),
                "text_config_batch": ("TEXT_CONFIG_BATCH",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, threshold, torchscript_jit, output_type, background_mode,
                          background_color="#000000", gradient_color1="#000000", gradient_color2="#FFFFFF",
                          gradient_direction="horizontal", background_image=None, text_config_batch=None):

        img_list = []
        mask_list = []

        for img_idx, img in enumerate(tqdm(image, "Processing Images")):
            pil_img = tensor2pil(img)
            original_size = pil_img.size

            if output_type == "original":
                rgba_img = pil_img.convert('RGBA')
                processed_output = rgba_img
                alpha_mask = np.ones(
                    (original_size[1], original_size[0]), dtype=np.float32)
            else:
                if (torchscript_jit == "default"):
                    remover = Remover()
                else:
                    remover = Remover(jit=True)

                rgba_output = remover.process(
                    pil_img, type='rgba', threshold=threshold)
                rgba_array = np.array(rgba_output)
                alpha_mask = rgba_array[:, :, 3] / 255.0

                if output_type == "replace_background":
                    if background_mode == "color":
                        bg_color = hex_to_rgb(background_color)
                        new_bg = Image.new(
                            'RGBA', original_size, (*bg_color, 255))
                    elif background_mode == "gradient":
                        color1 = hex_to_rgb(gradient_color1)
                        color2 = hex_to_rgb(gradient_color2)
                        new_bg = create_gradient_background(
                            original_size, color1, color2, gradient_direction)
                    elif background_mode == "image" and background_image is not None:
                        bg_idx = min(img_idx, len(background_image) - 1)
                        bg_pil = tensor2pil(background_image[bg_idx])
                        new_bg = bg_pil.resize(original_size).convert('RGBA')
                    else:
                        new_bg = Image.new(
                            'RGBA', original_size, (0, 0, 0, 255))

                    # Apply background text if configured
                    if text_config_batch:
                        for text_config in text_config_batch:
                            if text_config["text_placement"] in ["background", "both"]:
                                new_bg = add_text_overlay(new_bg, text_config)

                    processed_output = Image.alpha_composite(
                        new_bg, rgba_output)

                    # Apply foreground text if configured
                    if text_config_batch:
                        for text_config in text_config_batch:
                            if text_config["text_placement"] in ["foreground", "both"]:
                                processed_output = add_text_overlay(
                                    processed_output, text_config)

                elif output_type == "background":
                    orig_rgba = np.concatenate(
                        [np.array(pil_img), np.ones_like(alpha_mask)[..., None] * 255], axis=-1)
                    inverse_mask = 1 - alpha_mask
                    bg_rgba = orig_rgba.copy()
                    bg_rgba[:, :, 3] = inverse_mask * 255
                    processed_output = Image.fromarray(
                        bg_rgba.astype(np.uint8))

                    # Apply background text if configured
                    if text_config_batch:
                        for text_config in text_config_batch:
                            if text_config["text_placement"] in ["background", "both"]:
                                processed_output = add_text_overlay(
                                    processed_output, text_config)

                else:  # foreground
                    processed_output = rgba_output
                    # Apply foreground text if configured
                    if text_config_batch:
                        for text_config in text_config_batch:
                            if text_config["text_placement"] in ["foreground", "both"]:
                                processed_output = add_text_overlay(
                                    processed_output, text_config)

            out = pil2tensor(processed_output)
            img_list.append(out)

            mask_tensor = torch.from_numpy(alpha_mask).unsqueeze(0)
            mask_list.append(mask_tensor)

        img_stack = torch.cat(img_list, dim=0)
        mask_stack = torch.cat(mask_list, dim=0)

        if output_type == "background":
            mask_stack = 1 - mask_stack

        return (img_stack, mask_stack)
