DEFINING_FEATURES = {
    "toaster": [
        "2-slot",
        "4-slot",
        "rounded retro",
        "sleek modern",
        "blocky industrial",
    ],
    "hair drier": [
        "L-shaped",
        "conical",
        "spherical vintage",
        "compact foldable", 
        "diffuser attachment",
    ]
}

MATERIALS = {
    "toaster": [
        "", 
        "brushed stainless steel",     
        "cream enamel",
        "mirror chrome",
        "matte black",
        "white plastic",
        "brushed copper",
        "anodized aluminum",
        "mint green plastic",
        "glass side-panels"
    ],
    "hair drier": [
        "", 
        "matte black polycarbonate",
        "glossy ABS plastic",
        "rubberized",
        "metallic rose gold",
        "pearlized ceramic",
        "transparent acrylic",
        "satin chrome",
        "carbon fiber"
    ],
}

CONTEXTS = {
    "toaster": [
        "kitchen counter",
        "wooden table",
        "stone surface",
        "metal workbench",
        "black surface",
        "floating void",
        "diner table",
        "pedestal",
        ""
    ],
    "hair drier": [
        "green bathroom counter",
        "red vanity table",
        "white surface",
        "metal workbench",
        "wooden surface",
        "salon station",
        ""
    ]
}

LIGHTING = [
    "studio light",
    "soft daylight",
    "dramatic side-light",
    "flat daylight",
    "bright daylight",
    "golden hour",
    "overcast",
    "harsh sunlight",
    "low-key",
    "",
    "",
]

FRAMING = [
    "angled",
    "profile",
    "close-up",
    "asymmetric",
    "low angle",
    "", 
    "",
]

SHOT_TYPES = [
    "street photo",
    "snapshot",
    "instagram",
    "polaroid",
    "35mm film",
    "vintage photo",
    "iPhone photo",
    "disposable camera",
    "flash photography",
    "cctv",
    "",
    ""
]

QUALITY_TAGS = [
    "blurry",
    "realistic",
    "grainy",
    "heavy grain",
    "casual",
    "faded warm",
    "overexposed",
    "cool tones",
    "pixelated",
    "",
    "",
]

_BASE_NEG = "text, letters, watermark, out of frame, cropped, logo"
NEGATIVE_PROMPTS = {
    "toaster": f"{_BASE_NEG}, floating slots, person, hands",
    "hair drier": f"{_BASE_NEG}, electric razor, straightener, curling iron, toothbrush, gun, drill, power tool"
}
