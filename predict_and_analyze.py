import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import pandas as pd
from datetime import datetime, timedelta

# --- New Imports for Weather Forecasting ---
import openmeteo_requests
import requests_cache
from retry_requests import retry
import geocoder

# --- Import for the Sarvam AI Library ---
from sarvamai import SarvamAI

# --- ZoneInfo Handling for different Python versions ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = pytz.timezone

# Import the U-Net model from the other file
from unet_model import UNet

# --- Configuration ---
PIXELS_PER_METER = 350.0
CONTAINER_SIZES = {'small': 0.09, 'medium': 0.25, 'large': 0.64, 'extra_large': 1.0}
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Gardening Assistant Class (Updated) ---
class GardeningAssistant:
    """
    Uses Sarvam AI to first generate a guide in English,
    and then translate it into the user's chosen language.
    """
    def __init__(self, language='english'):
        self.language = language
        self.lang_code_map = {
            'english': 'en-IN', 'hindi': 'hi-IN', 'punjabi': 'pa-IN',
            'marathi': 'mr-IN', 'gujarati': 'gu-IN', 'tamil': 'ta-IN',
            'telugu': 'te-IN', 'bengali': 'bn-IN'
        }
        try:
            # WARNING: Storing API keys in code is NOT secure. Do not share this file publicly.
            # PASTE YOUR SARVAM AI API KEY HERE:
            api_key = "sk_g7en0w4w_Fb2oFKkOEysqXd3IzoeXtDrm"

            if api_key == "YOUR_SARVAM_AI_API_KEY_HERE":
                print("\n" + "="*60)
                print("❌ FATAL ERROR: Sarvam AI API Key not set in the code.")
                print("Please replace 'YOUR_SARVAM_AI_API_KEY_HERE' with your actual API key.")
                print("Get a key from Sarvam AI: https://www.sarvam.ai/")
                print("="*60)
                sys.exit(1)

            self.client = SarvamAI(api_subscription_key=api_key)
            print(f"\n✔️ Gardening Assistant (Sarvam AI) is ready. Target Language: {self.language.capitalize()}")
        except Exception as e:
            print(f"❌ Error initializing Sarvam AI: {e}")
            self.client = None

    def generate_planting_guide(self, recommended_plants):
        if not self.client or not recommended_plants:
            return

        print("\n" + "=" * 50)
        print("🤖 Generating personalized step-by-step planting guide in English (for best quality)...")

        plant_names = [plant['plant_name'] for plant in recommended_plants]
        plant_list_str = ", ".join(plant_names)

        prompt = f"""
        You are a friendly and expert urban gardening assistant for Indian gardeners.
        Your task is to create a detailed, step-by-step planting guide in English for a beginner who wants to grow the following plants in containers on their balcony: {plant_list_str}.

        For each plant, provide the following:
        1. A short, encouraging introduction to the plant.
        2. A "Products You'll Need" section, listing essential items.
        3. A "Step-by-Step Planting Guide" with clear, numbered instructions.
        4. A "Quick Care Tips" section with 2-3 bullet points on basic care.

        At the very end of the entire response, add the promotional message:
        "Find all the high-quality seeds, soil, pots, and tools you need to get started at [Your Website Link Here]!"

        Structure the output using Markdown. Use headings for each plant name and separate each plant's entire section from the next with '---'.
        """

        try:
            # STEP 1: Generate the guide in English
            response = self.client.chat.completions(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            english_guide = response.choices[0].message.content
            print("✔️ English guide generated successfully!")

            # STEP 2: Translate the guide if needed
            if self.language == 'english':
                print("=" * 50 + "\n")
                print(english_guide)
            else:
                print(f"Translating guide to {self.language.capitalize()}...")
                
                text_chunks = english_guide.split('---')
                translated_chunks = []
                
                for i, chunk in enumerate(text_chunks):
                    if chunk.strip():
                        print(f"  - Translating chunk {i+1}/{len(text_chunks)}...")
                        target_code = self.lang_code_map.get(self.language, 'en-IN')
                        
                        translate_response = self.client.text.translate(
                            input=chunk,
                            source_language_code="auto",
                            target_language_code=target_code
                        )
                        
                        # --- FIX: Convert the entire response object to a string ---
                        translated_chunks.append(str(translate_response))
                
                final_guide = "\n---\n".join(translated_chunks)
                
                print("✔️ Guide translated successfully!")
                print("=" * 50 + "\n")
                print(final_guide)

        except Exception as e:
            print(f"❌ Could not complete AI task. Error: {e}")


# --- Recommendation Engine Class (Unchanged) ---
class RecommendationEngine:
    def __init__(self, db_path="indian_urban_plants_space_aware.csv"):
        try:
            self.plant_db = pd.read_csv(db_path)
        except FileNotFoundError:
            print(f"❌ Error: Plant database '{db_path}' not found.")
            self.plant_db = None

    def _parse_sunlight(self, sunlight_str):
        if '6+' in sunlight_str: return 6.0
        if '4-6' in sunlight_str: return 4.0
        return 0.0

    def recommend(self, forecast_df, space_analysis):
        if self.plant_db is None or forecast_df.empty or space_analysis is None:
            return []
        avg_min_temp = forecast_df['temp_min_c'].mean()
        avg_max_temp = forecast_df['temp_max_c'].mean()
        avg_sunshine = forecast_df['sunshine_hours'].mean()
        print("\n--- ☀️ Weekly Weather Summary ---")
        print(f"Avg Temp Range: {avg_min_temp:.1f}°C - {avg_max_temp:.1f}°C")
        print(f"Avg Daily Sunshine: {avg_sunshine:.1f} hours")
        print("--------------------------------\n")
        container_capacity = space_analysis['container_capacity']
        recommendations = []
        for index, plant in self.plant_db.iterrows():
            if not (plant['min_temp_c'] <= avg_min_temp and plant['max_temp_c'] >= avg_max_temp):
                continue
            required_sun = self._parse_sunlight(plant['sunlight_hours'])
            if avg_sunshine < required_sun:
                continue
            required_pot_size = plant['container_size'].lower()
            if container_capacity.get(required_pot_size, 0) <= 0:
                continue
            current_month = "August"
            is_monsoon = current_month in ["June", "July", "August", "September"]
            is_winter = current_month in ["November", "December", "January", "February"]
            is_summer = not (is_monsoon or is_winter)
            plant_seasons = plant['season'].lower()
            if "all year" not in plant_seasons:
                if (is_monsoon and "monsoon" not in plant_seasons) or \
                   (is_winter and "winter" not in plant_seasons) or \
                   (is_summer and "summer" not in plant_seasons):
                    continue
            recommendations.append(plant)
        sorted_recommendations = sorted(recommendations, key=lambda x: ["Easy", "Medium", "Hard"].index(x['difficulty']))
        self._display_recommendations(sorted_recommendations)
        return sorted_recommendations

    @staticmethod
    def _display_recommendations(recommendations):
        print("--- 🌱 Recommended Plants (Weather & Space Appropriate) ---")
        if not recommendations:
            print("No ideal plants found for the upcoming weather and your available space.")
        else:
            for i, plant in enumerate(recommendations, 1):
                print(f"{i}. {plant['plant_name']}")
        print("\n" + "=" * 50)

# --- Weather Forecaster Class (Unchanged) ---
class WeatherForecaster:
    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.weather_codes = {0:"☀️ Clear",1:"🌤️ Mainly clear",2:"🌥️ Partly cloudy",3:"☁️ Overcast",45:"🌫️ Fog",61:"🌧️ Slight rain",63:"🌧️ Moderate rain",65:"🌧️ Heavy rain",80:"🌦️ Slight showers",81:"🌦️ Moderate showers",82:"🌦️ Violent showers",95:"⛈️ Thunderstorm"}

    def get_user_location(self):
        print("📍 Detecting your location for the weather forecast...")
        print("✔️ Location set to: Delhi, India")
        return [28.6139, 77.2090]

    def get_weather_forecast(self):
        latitude, longitude = self.get_user_location()
        try:
            ist_tz = ZoneInfo('Asia/Kolkata')
            now_ist = datetime(2025, 8, 17, 11, 26, 43, tzinfo=ist_tz)
        except Exception:
            ist_tz = pytz.timezone('Asia/Kolkata')
            now_ist = datetime(2025, 8, 17, 11, 26, 43, tzinfo=ist_tz)
            
        today = now_ist.strftime('%Y-%m-%d')
        end_date = (now_ist + timedelta(days=6)).strftime('%Y-%m-%d')
        print(f"🗓️ Fetching forecast from {today} to {end_date}")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude":latitude,"longitude":longitude,"daily":["weather_code","temperature_2m_max","temperature_2m_min","sunshine_duration","precipitation_sum","precipitation_probability_max"],"timezone":"Asia/Kolkata","start_date":today,"end_date":end_date}
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            dates = pd.date_range(start=pd.to_datetime(daily.Time(), unit="s", utc=True),end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),freq=pd.Timedelta(seconds=daily.Interval()),inclusive="left").tz_convert('Asia/Kolkata')
            daily_data = {"date":dates,"weather":[self.weather_codes.get(code, "Unknown") for code in daily.Variables(0).ValuesAsNumpy()],"temp_max_c":daily.Variables(1).ValuesAsNumpy(),"temp_min_c":daily.Variables(2).ValuesAsNumpy(),"sunshine_hours":daily.Variables(3).ValuesAsNumpy()/3600,"precipitation_mm":daily.Variables(4).ValuesAsNumpy(),"rain_chance_%":daily.Variables(5).ValuesAsNumpy()}
            df = pd.DataFrame(data=daily_data)
            self.display_forecast(df, now_ist)
            return df
        except Exception as e:
            print(f"\n❌ Could not fetch weather data. Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def display_forecast(df, current_time_ist):
        print("\n--- 🌿 7-Day Garden Weather Forecast ---")
        print(f"📅 Forecast generated on: {current_time_ist.strftime('%A, %B %d, %Y at %I:%M %p IST')}")
        print("=" * 50)
        for _, row in df.iterrows():
            day_name = row['date'].strftime('%A, %b %d')
            print(f"\n📅 {day_name}: {row['weather']}\n" f"  🌡️ Temp: {row['temp_min_c']:.1f}°C to {row['temp_max_c']:.1f}°C\n" f"  ☀️ Sunshine: {row['sunshine_hours']:.1f} hrs\n" f"  💧 Rain: {row['precipitation_mm']:.1f} mm ({row['rain_chance_%']:.0f}% chance)")
        print("\n" + "=" * 50)

# --- Space Analyzer Class (Unchanged) ---
class SpaceAnalyzer:
    @staticmethod
    def analyze_growing_area(binary_mask, pixels_per_meter=PIXELS_PER_METER):
        if cv2.countNonZero(binary_mask) == 0: return None
        total_pixels = cv2.countNonZero(binary_mask)
        total_area_sqm = total_pixels / (pixels_per_meter ** 2)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        width_m = w / pixels_per_meter
        height_m = h / pixels_per_meter
        container_capacity = SpaceAnalyzer.calculate_container_capacity(total_area_sqm)
        return {'total_area_sqm':round(total_area_sqm,3),'width_meters':round(width_m,2),'height_meters':round(height_m,2),'container_capacity':container_capacity}

    @staticmethod
    def calculate_container_capacity(total_area_sqm):
        capacity = {}
        usable_area = total_area_sqm * 0.8
        for size_name, size_area in CONTAINER_SIZES.items():
            capacity[size_name] = max(0, int(usable_area // size_area))
        return capacity

# --- Main Prediction Function (Unchanged) ---
def predict(model, device, image_path, output_dir):
    original_image = Image.open(image_path).convert("RGB")
    original_w, original_h = original_image.size
    transform = T.Compose([T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), T.ToTensor(),])
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    print("🤖 Model is making a prediction...")
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = (torch.sigmoid(output) > 0.5).float()
    mask_np = predicted_mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    full_size_mask = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    binary_mask_for_analysis = (full_size_mask * 255).astype(np.uint8)
    print("📐 Analyzing the predicted area...")
    analysis_results = SpaceAnalyzer.analyze_growing_area(binary_mask_for_analysis)
    
    if analysis_results:
        print("\n--- ✅ Analysis Complete ---")
        print(f"Total Usable Area: {analysis_results['total_area_sqm']} sq. meters")
        print(f"Dimensions (Approx): {analysis_results['width_meters']}m x {analysis_results['height_meters']}m")
        print("Container Capacity:")
        for size, count in analysis_results['container_capacity'].items(): print(f"  - {size.capitalize()}: {count} pot(s)")
        print("-------------------------\n")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    overlay_save_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    original_cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    green_overlay = np.zeros_like(original_cv_image, dtype=np.uint8)
    green_overlay[:, :, 1] = 255
    mask_indices = binary_mask_for_analysis == 255
    blended_image = original_cv_image.copy()
    blended_image[mask_indices] = cv2.addWeighted(original_cv_image, 0.4, green_overlay, 0.6, 0)[mask_indices]
    cv2.imwrite(overlay_save_path, blended_image)
    print(f"✔️ Overlay image saved to: {overlay_save_path}")
    cv2.imshow("Prediction Overlay", blended_image)
    cv2.waitKey(1)
    
    return analysis_results

# --- MAIN EXECUTION BLOCK (Unchanged) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze space, forecast weather, and recommend plants with multi-lingual AI guides.")
    parser.add_argument("--image", type=str, required=True, help="Path to the new balcony image.")
    parser.add_argument("--model-path", type=str, default="unet_model_output/unet_balcony_best.pth", help="Path to the trained .pth model file.")
    parser.add_argument(
        "--language", 
        type=str, 
        default="english", 
        choices=['english', 'hindi', 'punjabi', 'marathi', 'gujarati', 'tamil', 'telugu', 'bengali'],
        help="Language for the generated gardening guide."
    )
    args = parser.parse_args()

    plant_db_filename = "indian_urban_plants_space_aware.csv"
    if not os.path.exists(plant_db_filename):
        print(f"FATAL ERROR: The plant database '{plant_db_filename}' was not found.")
        sys.exit(1)

    print("\n🖼️ Starting Image Analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    output_directory = "predictions"
    os.makedirs(output_directory, exist_ok=True)
    space_analysis_results = predict(model, device, args.image, output_directory)

    print("\n🌤️ Starting Weather Forecast Analysis...")
    forecaster = WeatherForecaster()
    forecast_df = forecaster.get_weather_forecast()

    recommended_plants = [] 
    if forecast_df is not None and not forecast_df.empty and space_analysis_results is not None:
        engine = RecommendationEngine(db_path=plant_db_filename)
        recommended_plants = engine.recommend(forecast_df, space_analysis_results)
    else:
        print("\nCould not generate recommendations due to missing weather or space data.")

    if recommended_plants:
        assistant = GardeningAssistant(language=args.language)
        assistant.generate_planting_guide(recommended_plants)
    else:
        print("\nAs no plants were recommended, no planting guide will be generated.")
    
    print("\nPress any key in the 'Prediction Overlay' window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nScript finished.")