import random
import csv

# Classes
classes = [
    "empty rooftop",
    "empty balcony",
    "empty window sill",
]

# Variables
weather_options = [
    "sunny", "cloudy", "rainy", "hazy", "overcast",
    "golden hour", "foggy", "nighttime with street lights"
]

lighting_options = [
    "harsh sunlight", "soft daylight", "backlit", 
    "low light", "mixed lighting"
]

surface_conditions = [
    "faded concrete floor", "cracked tiles", "dusty cement floor", 
    "moss patches", "wet surface after rain", "slightly stained tiles",
    "weathered bricks", "mild dust accumulation"
]

background_options = [
    "city skyline", "distant hills", "water tanks", 
    "clotheslines", "television antennas", "neighbouring apartment buildings",
    "empty sky", "construction site in the distance"
]

view_angles = [
    "overhead view", "side view", "low angle", 
    "wide shot", "close-up"
]

# Generate prompts
prompts = []
per_class = 200

for place in classes:
    for _ in range(per_class):
        weather = random.choice(weather_options)
        lighting = random.choice(lighting_options)
        surface = random.choice(surface_conditions)
        background = random.choice(background_options)
        angle = random.choice(view_angles)

        prompt = (
            f"An {place} in India, {surface}, {angle}, "
            f"{weather} day, {lighting}, {background} in the background, realistic photograph."
        )

        prompts.append({"class": place.replace(" ", "_"), "prompt": prompt})

# Save to CSV
with open("empty_places_prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "prompt"])
    writer.writeheader()
    writer.writerows(prompts)

print(f"Generated {len(prompts)} prompts and saved to empty_places_prompts.csv")