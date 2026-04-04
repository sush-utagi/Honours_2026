import json
import random
from pathlib import Path

TARGET_CLASSES = ["toaster", "hair drier", "toothbrush", "backpack", "parking meter", "spoon"]

TI_TOKENS = {
    "toaster": "<coco-toaster>",
    "hair drier": "<coco-dryer>",
    "toothbrush": "<coco-toothbrush>",
    "backpack": "<coco-backpack>",
    "parking meter": "<coco-meter>",
    "spoon": "<coco-spoon>"
}

DEFINING_FEATURES = {
    "toaster": [
        "(visible slots, lever:1.2), classic rectangular boxy shape",
        "(visible slots, lever:1.2), retro rounded dome silhouette",
        "(visible slots, lever:1.2), futuristic sleek curved design",
        "(visible slots, lever:1.2), chunky industrial square shape",
        "(visible slots, lever:1.2), ultra-thin minimalist profile"
    ],
    "hair drier": [
        "(nozzle, ergonomic blowdryer handle:1.2), air intake vent, coiled power cord, L-shaped silhouette",
        "(nozzle, ergonomic blowdryer handle:1.2), air intake vent, sleek conical modern design, power cable",
        "(nozzle, ergonomic blowdryer handle:1.2), air intake vent, vintage spherical body, electric cord",
        "(nozzle, ergonomic blowdryer handle:1.2), air intake vent, compact folding salon design",
        "(nozzle, ergonomic blowdryer handle:1.2), air intake vent, elongated futuristic cylindrical body"
    ],
    "toothbrush": [
        "(bristled head, long slender handle:1.2), ergonomic rubberized grip",
        "(bristled head, long slender handle:1.2), sleek minimalist straight body",
        "(bristled head, long slender handle:1.2), thick electric base with charging port",
        "(bristled head, long slender handle:1.2), curved angled neck design",
        "(bristled head, long slender handle:1.2), vintage flat profile"
    ],
    "backpack": [
        "(shoulder straps, zippered compartments:1.2), classic schoolbag silhouette",
        "(shoulder straps, zippered compartments:1.2), rugged hiking rucksack design",
        "(shoulder straps, zippered compartments:1.2), sleek urban tech daypack",
        "(shoulder straps, zippered compartments:1.2), soft vintage canvas knapsack",
        "(shoulder straps, top handle, zippered compartments:1.2), rigid aerodynamic hardshell"
    ],
    "parking meter": [
        "(sturdy vertical pole, coin slot, digital display screen:1.2), modern rectangular domed top",
        "(sturdy vertical pole, coin slot, time dial indicator:1.2), classic retro twin-headed design",
        "(sturdy vertical pole, coin slot, time dial indicator:1.2), heavy vintage industrial silhouette",
        "(sturdy vertical pole, card reader, solar panel:1.2), sleek smart-city kiosk shape",
        "(sturdy vertical pole, coin slot, digital display screen:1.2), weathered industrial cylindrical head"
    ],
    "spoon": [
        "(concave oval bowl, long curved handle:1.2), classic teardrop silhouette",
        "(concave oval bowl, long curved handle:1.2), ornate vintage engraved stem",
        "(concave oval bowl, long curved handle:1.2), modern minimalist flat profile",
        "(concave oval bowl, long curved handle:1.2), deep soup ladle shape",
        "(concave oval bowl, long curved handle:1.2), chunky rustic carved design"
    ]
}

MATERIALS = {
    "toaster": [
        "", # Default
        "brushed stainless steel", "matte pastel plastic", 
        "weathered rusted iron", "scratched black polycarbonate",
        "polished chrome", "translucent frosted glass"
    ],
    "hair drier": [
        "",
        "matte pastel plastic", "scratched black polycarbonate", 
        "glossy white ceramic", "brushed aluminum",
        "translucent brightly colored plastic", "carbon fiber weave"
    ],
    "toothbrush": [
        "",
        "matte pastel plastic", "biodegradable bamboo", 
        "flexible silicone", "translucent frosted glass",
        "glossy resin"
    ],
    "backpack": [
        "",
        "heavy duty canvas fabric", "faded blue denim", 
        "glossy waterproof nylon", "distressed brown leather",
        "woven hemp", "weather-resistant polyurethane"
    ],
    "parking meter": [
        "",
        "weathered rusted iron", "brushed stainless steel", 
        "scratched black polycarbonate", "heavy cast iron",
        "faded yellow painted metal", "dented steel"
    ],
    "spoon": [
        "",
        "brushed stainless steel", "polished shiny silver", 
        "carved rustic wood", "matte pastel plastic", 
        "tarnished copper", "polished mahogany wood"
    ]
}

STYLES = [
    "studio lighting, minimalist background", 
    "macro photography",
    "cinematic product shot", 
    "octane render", 
    "industrial design style",
    "soft natural morning light"
]

CONTEXTS = {
    "toaster": [
        "placed on a kitchen island", "set on a laboratory workbench", "displayed on a stone pedestal", 
        "resting on a table in a sun-drenched room", "positioned among abstract shapes", "floating in a white void",
        "sitting on a workbench", "placed on a cart in a high-tech server room", "resting on a picnic blanket", 
        "placed on the table of a retro diner booth", "resting on a stack of blueprints", 
        "displayed on a pedestal in a minimalist art gallery", "sitting on a dusty attic floor"
    ],
    "hair drier": [
        "placed on a marble vanity", "resting on a counter in a futuristic salon", "set on a glass shelf", 
        "placed on a table in a dressing room", "resting on a metallic surface", "floating among bubbles and mist",
        "set on a table in an industrial studio", "resting on a backstage makeup table", "placed inside a luxury gym locker", 
        "resting on a velvet cushion", "set on a counter in a neon-lit cyberpunk barbershop", 
        "placed next to a professional camera rig", "resting on a sleek hotel bathroom counter"
    ],
    "toothbrush": [
        "standing in a glass cup", "resting on a ceramic bathroom sink", "placed on a damp shower shelf", 
        "set on a metal tray in a sterile dentist office", "resting on a pristine white towel", "floating in zero gravity", 
        "placed next to a tube of toothpaste", "packed inside a travel toiletry bag", "resting on a bedside nightstand", 
        "set on a counter in a sunlit spa", "placed on a bright turquoise tile", 
        "resting next to a designer perfume bottle", "set on a clean bamboo mat"
    ],
    "backpack": [
        "resting on a wooden school desk", "leaning against a forest pine tree", "sitting on a bustling subway seat", 
        "placed on an airport luggage carousel", "hanging from a metal coat rack", "resting on top of a rocky mountain peak", 
        "placed inside an empty locker", "dropped on a rainy cobblestone street", "slumped against a library bookshelf", 
        "placed inside a bright yellow tent", "sitting on a high-speed train platform", "strapped to the back of a motorbike", 
        "thrown on a chair in a chaotic artist's studio"
    ],
    "parking meter": [
        "installed on a bustling city sidewalk", "standing covered in a light dusting of snow", 
        "mounted on a cracked asphalt street corner", "standing bathed in golden hour sunlight", 
        "installed next to a parallel parked car", "standing glowing under a street lamp at night", 
        "installed in an empty parking lot", "standing on a foggy coastal road", 
        "installed outside a vintage cinema", "standing in a futuristic utopia with flying cars", 
        "installed surrounded by autumn leaves", "standing on a rain-slicked boulevard", 
        "installed at a desert rest stop"
    ],
    "spoon": [
        "resting in a ceramic soup bowl", "placed on a textured linen napkin", "set next to a cup of steaming coffee", 
        "resting on a rustic wooden dining table", "floating in a splash of milk", "placed on a shiny metallic restaurant counter", 
        "displayed illuminated in a dark moody studio", "resting in a pile of colorful spices", 
        "stuck inside a honey jar", "placed on a gold-rimmed porcelain plate", 
        "resting on a velvet tablecloth", "set on a messy chef's workstation", 
        "packed inside a child's colorful lunchbox"
    ]
}
QUALITY_MODIFIERS = [
    "high quality, masterpiece, 8k", 
    "amateur photo, grainy, accidental shot", 
    "low resolution, CCTV footage style", 
    "sharp focus, professional photography",
    "faded polaroid, vintage aesthetic",
    "blurry background, real life photo",
    ""  # no quality modifier in some prompts
]

FRAMING_MODIFIERS = [
    "full view, showing entire object", # The "Standard" crop
    "close-up, focus on central details", # High-detail zoom
    "slightly tilted, dynamic angle",     # Rotation invariance
    "top-down view",            # Perspective variety
    "eye-level, front-on view",           # Prototypical view
    "partially cropped at edges"          # Robustness to tight bounding boxes
]

def generate_and_save_class_jsons(classes, num_per_class=100):
    cwd = Path.cwd()
    
    for cls in classes:
        samples_list = []
        ti_token = TI_TOKENS.get(cls, cls)
        neg_prompt = "logo, watermark"
        token_name = ti_token.strip("<>").replace("coco-", "")
        embedding_path = f"data_generation_backend/embeddings/{token_name}/learned_embeds.safetensors"
        
        for _ in range(num_per_class):
                    material = random.choice(MATERIALS[cls])
                    context = random.choice(CONTEXTS[cls])
                    style = random.choice(STYLES)
                    features = random.choice(DEFINING_FEATURES[cls])
                    quality = random.choice(QUALITY_MODIFIERS)
                    framing = random.choice(FRAMING_MODIFIERS)
                    material_phrase = f"made of {material}" if material else ""
                    
                    prompt_parts = [
                        f"A ({ti_token}:1.3)",
                        features,
                        material_phrase,
                        context,
                        style,
                        quality,
                        framing,
                        "centered"
                    ]
                    
                    clean_prompt = ", ".join([part for part in prompt_parts if part]) + "."
                    print(f"Generated prompt for {cls}: {clean_prompt}")
                    samples_list.append({
                        "prompt": clean_prompt,
                        "negative_prompt": neg_prompt,
                        "cfg_scale": random.randint(7, 12) # Bumped up to enforce the new prompt structure
                    })        
        final_data = {
            "coco_class": cls,
            "ti_token": ti_token,
            "embedding_path": embedding_path,
            "samples": samples_list
        }
        
        filename = f"{cls.replace(' ', '_')}_prompts.json"
        with open(cwd / filename, "w") as f:
            json.dump(final_data, f, indent=4)
            
        print(f"[done] Saved {filename} with {num_per_class} samples.")

if __name__ == "__main__":
    generate_and_save_class_jsons(TARGET_CLASSES, num_per_class=100)