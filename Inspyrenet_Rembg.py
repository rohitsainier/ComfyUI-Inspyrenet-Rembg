from PIL import Image, ImageDraw
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
                "output_type": (["foreground", "background", "replace_background"],),
                "background_mode": (["color", "gradient", "image"],),
            },
            "optional": {
                # Hex color
                "background_color": ("STRING", {"default": "#000000"}),
                # Hex color
                "gradient_color1": ("STRING", {"default": "#000000"}),
                # Hex color
                "gradient_color2": ("STRING", {"default": "#FFFFFF"}),
                "gradient_direction": (["horizontal", "vertical"],),
                "background_image": ("IMAGE",),  # Optional background image
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, threshold, torchscript_jit, output_type, background_mode,
                          background_color="#000000", gradient_color1="#000000", gradient_color2="#FFFFFF",
                          gradient_direction="horizontal", background_image=None):
        if (torchscript_jit == "default"):
            remover = Remover()
        else:
            remover = Remover(jit=True)

        img_list = []
        mask_list = []

        for img_idx, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            # Convert to PIL and process
            pil_img = tensor2pil(img)
            original_size = pil_img.size

            # Get RGBA output
            rgba_output = remover.process(
                pil_img, type='rgba', threshold=threshold)
            rgba_array = np.array(rgba_output)

            # Extract alpha channel as mask
            alpha_mask = rgba_array[:, :, 3] / 255.0

            if output_type == "replace_background":
                # Create new background based on selected mode
                if background_mode == "color":
                    bg_color = hex_to_rgb(background_color)
                    new_bg = Image.new('RGBA', original_size, (*bg_color, 255))

                elif background_mode == "gradient":
                    color1 = hex_to_rgb(gradient_color1)
                    color2 = hex_to_rgb(gradient_color2)
                    new_bg = create_gradient_background(
                        original_size, color1, color2, gradient_direction)

                elif background_mode == "image" and background_image is not None:
                    # Use the corresponding background image or the first one if there's only one
                    bg_idx = min(img_idx, len(background_image) - 1)
                    bg_pil = tensor2pil(background_image[bg_idx])
                    new_bg = bg_pil.resize(original_size).convert('RGBA')
                else:
                    # Default to black if no valid background option is provided
                    new_bg = Image.new('RGBA', original_size, (0, 0, 0, 255))

                # Composite the foreground over the new background
                processed_output = Image.alpha_composite(new_bg, rgba_output)

            elif output_type == "background":
                # Create background with transparency
                orig_rgba = np.concatenate(
                    [np.array(pil_img), np.ones_like(alpha_mask)[..., None] * 255], axis=-1)
                inverse_mask = 1 - alpha_mask
                bg_rgba = orig_rgba.copy()
                bg_rgba[:, :, 3] = inverse_mask * 255
                processed_output = Image.fromarray(bg_rgba.astype(np.uint8))

            else:  # foreground
                processed_output = rgba_output

            # Convert to tensor and append
            out = pil2tensor(processed_output)
            img_list.append(out)

            # Store mask
            mask_tensor = torch.from_numpy(alpha_mask).unsqueeze(0)
            mask_list.append(mask_tensor)

        # Stack results
        img_stack = torch.cat(img_list, dim=0)
        mask_stack = torch.cat(mask_list, dim=0)

        if output_type == "background":
            mask_stack = 1 - mask_stack

        return (img_stack, mask_stack)
