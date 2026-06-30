"""
RAW adjustments parser and NumPy math helper.
Applies Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Temp, Tint, Saturation, and Vibrance.
"""

from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import numpy as np

def parse_xmp_adjustments(xmp_path: str) -> dict[str, float]:
    """Parse Lightroom-compatible crs adjustment sliders from an XMP sidecar file."""
    adjustments = {}
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'x': 'adobe:ns:meta/',
            'crs': 'http://adobe.com/camera-raw-settings/1.0/'
        }
        
        # 1. Search attributes on rdf:Description elements
        for desc in root.findall('.//rdf:Description', ns):
            for key, val in desc.attrib.items():
                local_key = key
                if '}' in key:
                    ns_uri, local_key = key.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in key:
                    prefix, local_key = key.split(':', 1)
                    if prefix != 'crs':
                        continue
                
                try:
                    adjustments[local_key] = float(val)
                except ValueError:
                    pass
                    
        # 2. Search child elements in crs namespace under rdf:Description
        for desc in root.findall('.//rdf:Description', ns):
            for child in desc:
                tag = child.tag
                local_key = tag
                if '}' in tag:
                    ns_uri, local_key = tag.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in tag:
                    prefix, local_key = tag.split(':', 1)
                    if prefix != 'crs':
                        continue
                if child.text:
                    try:
                        adjustments[local_key] = float(child.text.strip())
                    except ValueError:
                        pass
    except Exception:
        pass
    
    relevant_keys = {
        'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012',
        'Whites2012', 'Blacks2012', 'Temperature', 'Tint', 'Vibrance', 'Saturation'
    }
    return {k: v for k, v in adjustments.items() if k in relevant_keys}


def apply_adjustments_to_rgb(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Apply Exposure, Contrast, Highlights, Shadows, Temp, Tint, and Saturation transforms to the image."""
    if rgb_image is None or not adj:
        return rgb_image
        
    # Convert image to float32 normalized [0.0, 1.0] for calculations
    img = rgb_image.astype(np.float32) / 255.0
    
    # 1. Temperature & Tint
    temp_val = adj.get('Temperature', 0.0)
    tint_val = adj.get('Tint', 0.0)
    if temp_val > 1000.0:
        r_ref, g_ref, b_ref = _kelvin_to_rgb(6500.0)
        r_tgt, g_tgt, b_tgt = _kelvin_to_rgb(temp_val)
        r_scale = r_tgt / r_ref
        g_scale = g_tgt / g_ref
        b_scale = b_tgt / b_ref
        
        r_scale = 1.0 / (r_scale + 1e-5)
        g_scale = 1.0 / (g_scale + 1e-5)
        b_scale = 1.0 / (b_scale + 1e-5)
        
        img[:, :, 0] *= r_scale / g_scale
        img[:, :, 2] *= b_scale / g_scale
    elif temp_val != 0.0:
        scale = temp_val / 100.0
        img[:, :, 0] *= (1.0 + scale * 0.15)
        img[:, :, 2] *= (1.0 - scale * 0.15)
        
    if tint_val != 0.0:
        scale = tint_val / 100.0
        img[:, :, 1] *= (1.0 - scale * 0.15)

    # 2. Exposure
    exp_val = adj.get('Exposure2012', 0.0)
    if exp_val != 0.0:
        img *= (2.0 ** exp_val)
        
    img = np.clip(img, 0.0, 1.0)
        
    # 3. Contrast
    contrast_val = adj.get('Contrast2012', 0.0)
    if contrast_val != 0.0:
        c = contrast_val / 100.0
        factor = (259.0 * (c * 100.0 + 255.0)) / (255.0 * (259.0 - c * 100.0))
        img = factor * (img - 0.5) + 0.5
        img = np.clip(img, 0.0, 1.0)

    # 4. Highlights & Shadows
    hi_val = adj.get('Highlights2012', 0.0)
    sh_val = adj.get('Shadows2012', 0.0)
    if hi_val != 0.0 or sh_val != 0.0:
        if hi_val != 0.0:
            w_h = img * img
            img += w_h * (hi_val / 100.0) * 0.2
        if sh_val != 0.0:
            w_s = (1.0 - img) * (1.0 - img)
            img += w_s * (sh_val / 100.0) * 0.2
        img = np.clip(img, 0.0, 1.0)

    # 5. Whites & Blacks
    white_val = adj.get('Whites2012', 0.0)
    black_val = adj.get('Blacks2012', 0.0)
    if white_val != 0.0 or black_val != 0.0:
        if white_val != 0.0:
            w_w = img ** 4
            img += w_w * (white_val / 100.0) * 0.15
        if black_val != 0.0:
            w_b = (1.0 - img) ** 4
            img += w_b * (black_val / 100.0) * 0.15
        img = np.clip(img, 0.0, 1.0)

    # 6. Saturation & Vibrance
    sat_val = adj.get('Saturation', 0.0)
    vib_val = adj.get('Vibrance', 0.0)
    if sat_val != 0.0 or vib_val != 0.0:
        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        luma = np.expand_dims(luma, axis=-1)
        
        sat_scale = 1.0
        if sat_val != 0.0:
            sat_scale += sat_val / 100.0
            
        if vib_val != 0.0:
            max_val = np.max(img, axis=-1, keepdims=True)
            min_val = np.min(img, axis=-1, keepdims=True)
            s = (max_val - min_val) / (max_val + 1e-5)
            vib_factor = (vib_val / 100.0) * (1.0 - s)
            sat_scale += vib_factor
            
        img = luma + (img - luma) * sat_scale
        img = np.clip(img, 0.0, 1.0)

    return (img * 255.0).astype(np.uint8)


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    k = max(1000.0, min(40000.0, k))
    temp = k / 100.0
    
    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = temp - 60.0
        r = 329.698727446 * (r ** -0.1332047592)
        r = max(0.0, min(255.0, r))
        
    # Green
    if temp <= 66.0:
        g = temp
        g_val = max(1.0, g)
        g = 99.4708025861 * np.log(g_val) - 161.1195681661
        g = max(0.0, min(255.0, g))
    else:
        g = temp - 60.0
        g = 288.1221695283 * (g ** -0.0755148492)
        g = max(0.0, min(255.0, g))
        
    # Blue
    if temp >= 66.0:
        b = 255.0
    else:
        if temp <= 19.0:
            b = 0.0
        else:
            b = temp - 10.0
            b_val = max(1.0, b)
            b = 138.5177312231 * np.log(b_val) - 305.0447927307
            b = max(0.0, min(255.0, b))
            
    return r / 255.0, g / 255.0, b / 255.0
