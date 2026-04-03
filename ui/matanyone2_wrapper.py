
import gc
import os
import tqdm
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
import random
import cv2
from matanyone2.utils.device import get_default_device, safe_autocast_decorator

device = get_default_device()

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)

def _cleanup_memory():
    """Force memory cleanup for MPS (Apple Silicon) and CUDA devices."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
@safe_autocast_decorator()
def matanyone2(processor, frames_np, mask, r_erode=0, r_dilate=0, n_warmup=10, return_foreground=True):
    """
    Args:
        frames_np: [(H,W,C)]*n, uint8
        mask: (H,W), uint8
    Outputs:
        com: [(H,W,C)]*n, uint8
        pha: [(H,W,C)]*n, uint8
    """

    # print(f'===== [r_erode] {r_erode}; [r_dilate] {r_dilate} =====')
    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3)) if return_foreground else None
    objects = [1]

    # [optional] erode & dilate on given seg mask
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).to(device)

    frames_np = [frames_np[0]]* n_warmup + frames_np

    frames = [] if return_foreground else None
    phas = []
    for ti, frame_single in tqdm.tqdm(enumerate(frames_np)):
        image = to_tensor(frame_single).float().to(device)

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)      # encode given mask
            output_prob = processor.step(image, first_frame_pred=True)      # clear past memory for warmup frames
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)  # clear past memory for warmup frames
            else:
                output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)
        
        # DONOT save the warmup frames
        if ti > (n_warmup-1):
            pha = mask.unsqueeze(2).detach().to("cpu").numpy()
            phas.append((pha*255).astype(np.uint8))
            if return_foreground:
                com_np = frame_single / 255. * pha + bgr * (1 - pha)
                frames.append((com_np*255).astype(np.uint8))
        
        # Cleanup intermediate tensors and VRAM
        del image, output_prob
        if ti % 10 == 0:  # Cleanup every 10 frames to balance performance
            _cleanup_memory()
    
    # Final cleanup
    _cleanup_memory()
    
    return frames, phas


@torch.inference_mode()
@safe_autocast_decorator()
def matanyone2_streaming(processor, frames_np, mask, output_dir, r_erode=0, r_dilate=0, n_warmup=10, return_foreground=False):
    """
    Memory-efficient streaming version that saves frames directly to disk.
    Uses constant RAM regardless of video length.
    
    Args:
        processor: InferenceCore processor
        frames_np: [(H,W,C)]*n, uint8 - input frames
        mask: (H,W), uint8 - initial segmentation mask
        output_dir: str - directory to save alpha frames
        r_erode: int - erosion kernel size
        r_dilate: int - dilation kernel size  
        n_warmup: int - number of warmup frames
        return_foreground: bool - whether to also save foreground composites
    
    Returns:
        frame_count: int - number of frames processed
        alpha_paths: list - paths to saved alpha frames
        foreground_paths: list or None - paths to saved foreground frames
    """
    
    os.makedirs(output_dir, exist_ok=True)
    foreground_dir = os.path.join(output_dir, "foreground") if return_foreground else None
    if return_foreground:
        os.makedirs(foreground_dir, exist_ok=True)
    
    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3)) if return_foreground else None
    objects = [1]

    # [optional] erode & dilate on given seg mask
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).to(device)

    # Prepend warmup frames
    first_frame = frames_np[0]
    frames_np = [first_frame] * n_warmup + frames_np

    alpha_paths = []
    foreground_paths = [] if return_foreground else None
    frame_idx = 0
    
    for ti, frame_single in tqdm.tqdm(enumerate(frames_np), total=len(frames_np)):
        image = to_tensor(frame_single).float().to(device)

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)
        
        # DONOT save the warmup frames - save directly to disk
        if ti > (n_warmup - 1):
            pha = mask.unsqueeze(2).detach().to("cpu").numpy()
            alpha_frame = (pha * 255).astype(np.uint8)
            
            # Save alpha to disk immediately
            alpha_path = os.path.join(output_dir, f"{str(frame_idx).zfill(5)}.png")
            cv2.imwrite(alpha_path, alpha_frame[:, :, 0])
            alpha_paths.append(alpha_path)
            
            # Save foreground composite if requested
            if return_foreground:
                com_np = frame_single / 255. * pha + bgr * (1 - pha)
                foreground_frame = (com_np * 255).astype(np.uint8)
                foreground_path = os.path.join(foreground_dir, f"{str(frame_idx).zfill(5)}.png")
                cv2.imwrite(foreground_path, cv2.cvtColor(foreground_frame, cv2.COLOR_RGB2BGR))
                foreground_paths.append(foreground_path)
                del com_np, foreground_frame
            
            del pha, alpha_frame
            frame_idx += 1
        
        # Cleanup intermediate tensors and VRAM after each frame
        del image, output_prob
        
        # Force memory cleanup every frame for streaming mode
        _cleanup_memory()
    
    return frame_idx, alpha_paths, foreground_paths
