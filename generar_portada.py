from PIL import Image, ImageDraw, ImageFont
import os

# Configuración del canvas
width, height = 1920, 1080
bg_color = "#f0f2f5"
border_color = "#2c3e50"
section_color = "#3498db"
accent_color = "#e74c3c"
text_color = "#2c3e50"

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Crear imagen
img = Image.new('RGB', (width, height), hex_to_rgb(bg_color))
draw = ImageDraw.Draw(img)

# Dibujar borde decorativo
draw.rectangle([40, 40, width-40, height-40], outline=hex_to_rgb(border_color), width=4)
draw.rectangle([50, 50, width-50, height-50], outline=hex_to_rgb(section_color), width=2)

def get_font(size):
    # Intentar usar fuentes del sistema, fallback a default
    fonts_to_try = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    ]
    for font_path in fonts_to_try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

try:
    font_title = get_font(52)
    font_section = get_font(36)
    font_body = get_font(28)
    font_small = get_font(24)
except:
    font_title = ImageFont.load_default()
    font_section = ImageFont.load_default()
    font_body = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Título principal
title = "MatAnyone 2.1"
subtitle = "Optimized Video Matting for Apple Silicon"
y_pos = 80

draw.text((width//2, y_pos), title, fill=hex_to_rgb(text_color), font=font_title, anchor="mm")
y_pos += 70
draw.text((width//2, y_pos), subtitle, fill=hex_to_rgb(section_color), font=font_section, anchor="mm")
y_pos += 90

# Separador
draw.line([150, y_pos, width-150, y_pos], fill=hex_to_rgb(border_color), width=2)
y_pos += 50

# Sección: Características principales
features = [
    "- Disk-based Streaming: Solo 5 frames en RAM",
    "- VRAM Cleanup: gc.collect() + torch.mps.empty_cache()",
    "- Procesamiento de videos de cualquier duracion",
    "- Optimizado para Mac Studio / Apple Silicon"
]

section_title = "Optimizaciones Clave"
draw.text((100, y_pos), section_title, fill=hex_to_rgb(accent_color), font=font_section)
y_pos += 60

for feature in features:
    draw.text((120, y_pos), feature, fill=hex_to_rgb(text_color), font=font_body)
    y_pos += 45

y_pos += 30
draw.line([150, y_pos, width-150, y_pos], fill=hex_to_rgb(border_color), width=2)
y_pos += 50

# Sección: Créditos
credits_title = "Proyecto Original"
draw.text((100, y_pos), credits_title, fill=hex_to_rgb(accent_color), font=font_section)
y_pos += 60

original = [
    "MatAnyone 2 por Peiqing Yang, Shangchen Zhou, Kai Hao, Qingyi Tao",
    "S-Lab, NTU  |  Paper: arXiv:2512.11782"
]

for line in original:
    draw.text((120, y_pos), line, fill=hex_to_rgb(text_color), font=font_body)
    y_pos += 45

y_pos += 40
credits_fork = "Optimizaciones v2.1 (Este Fork)"
draw.text((100, y_pos), credits_fork, fill=hex_to_rgb(accent_color), font=font_section)
y_pos += 60

optimizations = [
    "Memory streaming, MPS cache cleanup, inference pipeline refactor",
    "github.com/emanuelbarriga/MatAnyone2.1"
]

for line in optimizations:
    draw.text((120, y_pos), line, fill=hex_to_rgb(text_color), font=font_body)
    y_pos += 45

# Footer
footer_text = "Videos de cualquier duración · RAM constante · Apple Silicon"
draw.text((width//2, height-60), footer_text, fill=hex_to_rgb(section_color), font=font_small, anchor="mm")

# Guardar
img.save('/Users/servermac/Documents/app/MatAnyone2/portada_matanyone21.png')
print("Imagen generada: /Users/servermac/Documents/app/MatAnyone2/portada_matanyone21.png")
print(f"Resolución: {width}x{height}px")
