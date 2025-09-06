import os
import sys
import time
import re
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import pandas as pd
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sarvamai import SarvamAI

# --- ZoneInfo Handling ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = pytz.timezone

from unet_model import UNet

# --- Configuration ---
PIXELS_PER_METER = 350.0
CONTAINER_SIZES = {'small': 0.09, 'medium': 0.25, 'large': 0.64, 'extra_large': 1.0}
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Gardening Assistant Class (Fixed with better error handling and retries) ---
class GardeningAssistant:
    def __init__(self, language='english'):
        self.language = language
        self.lang_code_map = {
            'english': 'en-IN', 'hindi': 'hi-IN', 'punjabi': 'pa-IN',
            'marathi': 'mr-IN', 'gujarati': 'gu-IN', 'tamil': 'ta-IN',
            'telugu': 'te-IN', 'bengali': 'bn-IN'
        }
        try:
            api_key = os.getenv("SARVAM_API_KEY", "sk_g7en0w4w_Fb2oFKkOEysqXd3IzoeXtDrm") # Replace with your key
            if not api_key or "YOUR_SARVAM_AI_API_KEY" in api_key:
                raise ValueError("Sarvam AI API Key not set.")
            # Increased timeout to 60 seconds and added retry logic
            self.client = SarvamAI(api_subscription_key=api_key, timeout=60.0)
        except Exception as e:
            print(f"❌ Error initializing Sarvam AI: {e}")
            self.client = None

    @staticmethod
    def _split_long_text(text, limit=800):  # Reduced limit for better handling
        """Split text into smaller chunks to avoid API limits"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for i, sentence in enumerate(sentences):
            if sentence:
                sentence_with_period = sentence + ("." if i < len(sentences) - 1 else "")
            else:
                continue

            if len(sentence_with_period) > limit:
                words = sentence_with_period.split(' ')
                word_chunk = ""
                for word in words:
                    if len(word_chunk) + len(word) + 1 <= limit:
                        word_chunk += word + " "
                    else:
                        if word_chunk.strip():
                            chunks.append(word_chunk.strip())
                        word_chunk = word + " "
                if word_chunk.strip():
                    chunks.append(word_chunk.strip())
                current_chunk = ""
            elif len(current_chunk) + len(sentence_with_period) + 1 <= limit:
                current_chunk += sentence_with_period + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence_with_period + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return [chunk for chunk in chunks if chunk.strip()]

    def _call_api_with_retry(self, api_call, max_retries=3):
        """Call API with retry logic for timeout handling"""
        for attempt in range(max_retries):
            try:
                print(f"  - API call attempt {attempt + 1}/{max_retries}")
                return api_call()
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(f"  - Timeout occurred, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  - All retry attempts failed due to timeout")
                        raise
                else:
                    print(f"  - Non-timeout error: {e}")
                    raise
        return None

    def generate_planting_guide(self, recommended_plants):
        if not self.client or not recommended_plants:
            return "Could not generate a planting guide at this time."

        plant_names = [plant['plant_name'] for plant in recommended_plants]
        plant_list_str = ", ".join(plant_names)
        
        # Create a shorter, more focused prompt to reduce API load
        prompt = f"""Create a concise planting guide for Indian balcony gardening with these plants: {plant_list_str}.

For each plant, provide:
1. Brief introduction
2. Essential supplies needed  
3. Simple planting steps
4. Basic care tips

Keep each section under 150 words. End with: "Find supplies at my-easy-garden.com!"
Use markdown formatting with plant names as headers."""
        
        try:
            print("Generating planting guide...")
            
            def generate_guide():
                return self.client.chat.completions(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500  # Reduced token limit
                )
            
            response = self._call_api_with_retry(generate_guide)
            if not response:
                return "Could not generate planting guide due to API timeout. Please try again later."
                
            english_guide = response.choices[0].message.content

            if self.language == 'english':
                return english_guide

            return self._translate_guide(english_guide)

        except Exception as e:
            print(f"❌ Error generating guide: {e}")
            return f"Error generating guide: {str(e)}. Please try again later."

    def _translate_guide(self, english_guide):
        """Translate guide with improved error handling and proper formatting"""
        print(f"Translating guide to {self.language.capitalize()}...")
        
        target_code = self.lang_code_map.get(self.language, 'en-IN')
        
        # Split guide into smaller, more manageable sections
        sections = english_guide.split('\n\n')  # Split by paragraphs instead of by '---'
        translated_sections = []

        for i, section in enumerate(sections):
            if not section.strip():
                translated_sections.append("")
                continue

            print(f"  - Translating section {i+1}/{len(sections)} ({len(section)} chars)")
            
            try:
                # Split section if too long
                if len(section) > 800:
                    chunks = self._split_long_text(section, limit=800)
                    print(f"    - Section split into {len(chunks)} chunks")
                else:
                    chunks = [section]

                translated_chunks = []
                for j, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    print(f"    - Translating chunk {j+1}/{len(chunks)}")
                    
                    def translate_chunk():
                        return self.client.text.translate(
                            input=chunk.strip(),
                            source_language_code="en-IN",  # More specific source language
                            target_language_code=target_code
                        )
                    
                    translate_response = self._call_api_with_retry(translate_chunk)
                    if not translate_response:
                        print(f"    - Translation failed for chunk {j+1}, using original text")
                        translated_chunks.append(chunk)
                        continue
                        
                    # Handle the response correctly
                    if hasattr(translate_response, 'translated_text'):
                        translated_text = translate_response.translated_text
                    elif hasattr(translate_response, 'text'):
                        translated_text = translate_response.text
                    else:
                        print(f"    - Unexpected response format, using original text")
                        translated_text = chunk
                    
                    translated_chunks.append(translated_text)
                    
                    # Add delay between API calls to avoid rate limiting
                    time.sleep(1.5)
                
                translated_sections.append(" ".join(translated_chunks))
                
            except Exception as e:
                print(f"    - Error translating section {i+1}: {e}")
                translated_sections.append(section)  # Fallback to original text
                continue

        final_guide = "\n\n".join(translated_sections)
        
        # Format the guide with proper line breaks for better readability
        formatted_guide = self._format_guide_with_line_breaks(final_guide)
        
        print("✔️ Translation completed!")
        return formatted_guide

    def _format_guide_with_line_breaks(self, guide_text):
        """Format the guide with proper line breaks for numbered points"""
        
        # Common patterns for numbered/bulleted lists in different languages
        numbered_patterns = [
            r'(\d+\.)',  # 1., 2., 3.
            r'(\d+\))',  # 1), 2), 3)
            r'(੧\.)', r'(੨\.)', r'(੩\.)', r'(੪\.)', r'(੫\.)',  # Punjabi numerals
            r'([•▪▫◦‣⁃])',  # Various bullet points
            r'(ਕਦਮ\s*\d+:)',  # Punjabi "Step 1:", "Step 2:", etc.
            r'(ਪਹਿਲਾਂ:)', r'(ਦੂਜਾ:)', r'(ਤੀਜਾ:)', r'(ਚੌਥਾ:)',  # Punjabi ordinals
        ]
        
        formatted_text = guide_text
        
        # Add line breaks before numbered points
        for pattern in numbered_patterns:
            formatted_text = re.sub(pattern, r'\n\1', formatted_text)
        
        # Clean up multiple newlines and ensure proper spacing
        formatted_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_text)
        formatted_text = re.sub(r'^\n+', '', formatted_text)  # Remove leading newlines
        
        # Add line breaks after colons in headings (for better structure)
        formatted_text = re.sub(r'([:#])\s*([ਕਦਮ|ਸਾਮਾਨ|ਦੇਖਭਾਲ])', r'\1\n\2', formatted_text)
        
        # Ensure each major section starts on a new line
        section_headers = [
            r'(ਸਾਮਾਨ\s*ਜੋ\s*ਤੁਹਾਨੂੰ\s*ਚਾਹੀਦਾ)',  # "Supplies you need"
            r'(ਕਦਮ-ਦਰ-ਕਦਮ)',  # "Step-by-step"
            r'(ਦੇਖਭਾਲ\s*ਦੇ\s*ਸੁਝਾਅ)',  # "Care tips"
            r'(ਤੇਜ਼\s*ਦੇਖਭਾਲ)',  # "Quick care"
            r'(ਉਤਪਾਦ\s*ਜੋ\s*ਤੁਹਾਨੂੰ\s*ਚਾਹੀਦੇ)',  # "Products you need"
        ]
        
        for header_pattern in section_headers:
            formatted_text = re.sub(header_pattern, r'\n\1', formatted_text)
        
        return formatted_text.strip()


# --- Recommendation Engine Class (Unchanged but with error handling) ---
class RecommendationEngine:
    def __init__(self, db_path="indian_urban_plants_space_aware.csv"):
        try:
            self.plant_db = pd.read_csv(db_path)
            print(f"✔️ Loaded plant database with {len(self.plant_db)} plants")
        except Exception as e:
            print(f"❌ Error loading plant database: {e}")
            self.plant_db = pd.DataFrame()

    def _parse_sunlight(self, sunlight_str):
        if pd.isna(sunlight_str):
            return 0.0
        sunlight_str = str(sunlight_str).lower()
        if '6+' in sunlight_str: 
            return 6.0
        if '4-6' in sunlight_str: 
            return 4.0
        if '2-4' in sunlight_str:
            return 2.0
        return 0.0

    def recommend(self, forecast_df, space_analysis):
        if self.plant_db.empty or forecast_df.empty:
            print("❌ No data available for recommendations")
            return []
            
        try:
            avg_min_temp = forecast_df['temp_min_c'].mean()
            avg_max_temp = forecast_df['temp_max_c'].mean()
            avg_sunshine = forecast_df['sunshine_hours'].mean()
            container_capacity = space_analysis.get('container_capacity', {})
            
            recommendations = []
            
            for index, plant in self.plant_db.iterrows():
                # Temperature check
                if not (plant['min_temp_c'] <= avg_min_temp and plant['max_temp_c'] >= avg_max_temp):
                    continue
                
                # Sunlight check
                required_sun = self._parse_sunlight(plant['sunlight_hours'])
                if avg_sunshine < required_sun:
                    continue
                
                # Container capacity check
                required_pot_size = str(plant['container_size']).lower()
                if container_capacity.get(required_pot_size, 0) <= 0:
                    continue
                
                # Seasonal check
                current_month = datetime.now().strftime("%B")
                is_monsoon = current_month in ["June", "July", "August", "September"]
                is_winter = current_month in ["November", "December", "January", "February"]
                is_summer = not (is_monsoon or is_winter)
                
                plant_seasons = str(plant['season']).lower()
                if "all year" not in plant_seasons:
                    if (is_monsoon and "monsoon" not in plant_seasons) or \
                       (is_winter and "winter" not in plant_seasons) or \
                       (is_summer and "summer" not in plant_seasons):
                        continue
                
                recommendations.append(plant.to_dict())
            
            # Sort by difficulty (Easy first)
            difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
            sorted_recommendations = sorted(
                recommendations, 
                key=lambda x: difficulty_order.get(str(x.get('difficulty', 'medium')).lower(), 1)
            )
            
            print(f"✔️ Found {len(sorted_recommendations)} suitable plants")
            return sorted_recommendations
            
        except Exception as e:
            print(f"❌ Error in recommendation engine: {e}")
            return []


# --- Weather Forecaster Class (Enhanced error handling) ---
class WeatherForecaster:
    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.weather_codes = {
            0:"☀️ Clear", 1:"🌤️ Mainly clear", 2:"🌥️ Partly cloudy", 3:"☁️ Overcast",
            45:"🌫️ Fog", 48:"🌫️ Depositing rime fog", 51:"🌦️ Light drizzle",
            53:"🌦️ Moderate drizzle", 55:"🌦️ Dense drizzle", 56:"🌨️ Light freezing drizzle",
            57:"🌨️ Dense freezing drizzle", 61:"🌧️ Slight rain", 63:"🌧️ Moderate rain",
            65:"🌧️ Heavy rain", 66:"🌨️ Light freezing rain", 67:"🌨️ Heavy freezing rain",
            71:"🌨️ Slight snow", 73:"🌨️ Moderate snow", 75:"🌨️ Heavy snow",
            77:"🌨️ Snow grains", 80:"🌦️ Slight showers", 81:"🌦️ Moderate showers",
            82:"🌦️ Violent showers", 85:"🌨️ Slight snow showers", 86:"🌨️ Heavy snow showers",
            95:"⛈️ Thunderstorm", 96:"⛈️ Thunderstorm with slight hail",
            99:"⛈️ Thunderstorm with heavy hail"
        }

    def get_weather_forecast(self, lat=28.6139, lon=77.2090):  # Default to Delhi
        try:
            ist_tz = ZoneInfo('Asia/Kolkata')
            now_ist = datetime.now(ist_tz)
            today = now_ist.strftime('%Y-%m-%d')
            end_date = (now_ist + timedelta(days=6)).strftime('%Y-%m-%d')
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": [
                    "weather_code", "temperature_2m_max", "temperature_2m_min",
                    "sunshine_duration", "precipitation_sum", "precipitation_probability_max"
                ],
                "timezone": "Asia/Kolkata",
                "start_date": today,
                "end_date": end_date
            }
            
            print("Fetching weather forecast...")
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            
            dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ).tz_convert('Asia/Kolkata')
            
            daily_data = {
                "date": dates.strftime('%A, %b %d').tolist(),
                "weather": [self.weather_codes.get(int(code), "Unknown") for code in daily.Variables(0).ValuesAsNumpy()],
                "temp_max_c": daily.Variables(1).ValuesAsNumpy().round(1),
                "temp_min_c": daily.Variables(2).ValuesAsNumpy().round(1),
                "sunshine_hours": (daily.Variables(3).ValuesAsNumpy() / 3600).round(1),
                "precipitation_mm": daily.Variables(4).ValuesAsNumpy().round(1),
                "rain_chance_%": daily.Variables(5).ValuesAsNumpy().round(0).astype(int)
            }
            
            df = pd.DataFrame(data=daily_data)
            forecast_summary = f"7-day forecast for Delhi, generated on: {now_ist.strftime('%A, %B %d, %Y at %I:%M %p IST')}"
            
            print("✔️ Weather forecast retrieved successfully")
            return df, forecast_summary
            
        except Exception as e:
            print(f"❌ Could not fetch weather data. Error: {e}")
            # Return empty dataframe with proper structure
            empty_df = pd.DataFrame({
                "date": [],
                "weather": [],
                "temp_max_c": [],
                "temp_min_c": [],
                "sunshine_hours": [],
                "precipitation_mm": [],
                "rain_chance_%": []
            })
            return empty_df, f"Could not fetch weather. Error: {e}"


# --- Space Analyzer Class (Enhanced with validation) ---
class SpaceAnalyzer:
    @staticmethod
    def analyze_growing_area(binary_mask, pixels_per_meter=PIXELS_PER_METER):
        try:
            if binary_mask is None or binary_mask.size == 0:
                print("❌ Invalid binary mask provided")
                return None
                
            total_pixels = cv2.countNonZero(binary_mask)
            if total_pixels == 0:
                print("❌ No growing area detected in the image")
                return None
            
            total_area_sqm = total_pixels / (pixels_per_meter ** 2)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("❌ No contours found in the mask")
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            width_m = w / pixels_per_meter
            height_m = h / pixels_per_meter
            
            container_capacity = SpaceAnalyzer.calculate_container_capacity(total_area_sqm)
            
            analysis_result = {
                'total_area_sqm': round(total_area_sqm, 3),
                'width_meters': round(width_m, 2),
                'height_meters': round(height_m, 2),
                'container_capacity': container_capacity
            }
            
            print(f"✔️ Space analysis completed: {total_area_sqm:.3f} sq meters")
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error in space analysis: {e}")
            return None

    @staticmethod
    def calculate_container_capacity(total_area_sqm):
        capacity = {}
        usable_area = total_area_sqm * 0.8  # 80% of space is usable
        
        for size_name, size_area in CONTAINER_SIZES.items():
            capacity[size_name] = max(0, int(usable_area // size_area))
        
        return capacity


# --- Enhanced Prediction Function ---
def predict(model, device, image_path, output_dir):
    try:
        print(f"Processing image: {image_path}")
        
        # Load and validate image
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None, None
            
        original_image = Image.open(image_path).convert("RGB")
        original_w, original_h = original_image.size
        print(f"  - Image dimensions: {original_w}x{original_h}")
        
        # Prepare input tensor
        transform = T.Compose([
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
        ])
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Run prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = (torch.sigmoid(output) > 0.5).float()
        
        # Process mask
        mask_np = predicted_mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        full_size_mask = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        binary_mask_for_analysis = (full_size_mask * 255).astype(np.uint8)
        
        # Analyze space
        analysis_results = SpaceAnalyzer.analyze_growing_area(binary_mask_for_analysis)
        if not analysis_results:
            print("❌ Could not analyze the space in the image")
            return None, None
        
        # Create overlay visualization
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        overlay_filename = f"{base_name}_{timestamp}_overlay.png"
        overlay_save_path = os.path.join(output_dir, overlay_filename)

        original_cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        green_overlay = np.zeros_like(original_cv_image, dtype=np.uint8)
        green_overlay[:, :, 1] = 255  # Green channel
        
        mask_indices = binary_mask_for_analysis.squeeze() == 255
        blended_image = original_cv_image.copy()
        blended_image[mask_indices] = cv2.addWeighted(
            original_cv_image, 0.4, green_overlay, 0.6, 0
        )[mask_indices]
        
        cv2.imwrite(overlay_save_path, blended_image)
        
        web_accessible_path = os.path.join('predictions', overlay_filename).replace("\\", "/")
        print(f"✔️ Overlay saved: {web_accessible_path}")
        
        return analysis_results, web_accessible_path
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, None


# --- Master Orchestration Function (Enhanced) ---
def run_full_analysis(image_path, language, model, device):
    """
    Run complete urban gardening analysis pipeline
    Returns dictionary with all analysis results
    """
    print(f"\n🌱 Starting full urban gardening analysis...")
    print(f"  - Image: {image_path}")
    print(f"  - Language: {language}")
    
    output_directory = "predictions"
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # 1. Space Analysis
        print("\n📏 Step 1: Analyzing growing space...")
        space_analysis_results, overlay_path = predict(model, device, image_path, output_directory)
        if not space_analysis_results:
            return {"error": "Could not analyze the image. Please ensure the image shows a clear balcony or growing space."}

        # 2. Weather Forecast  
        print("\n🌤️ Step 2: Getting weather forecast...")
        forecaster = WeatherForecaster()
        forecast_df, forecast_summary = forecaster.get_weather_forecast()
        if forecast_df.empty:
            return {"error": "Could not retrieve weather data. Please check your internet connection."}

        # 3. Plant Recommendations
        print("\n🌿 Step 3: Finding suitable plants...")
        engine = RecommendationEngine()
        recommended_plants = engine.recommend(forecast_df, space_analysis_results)
        
        # 4. AI Planting Guide
        print("\n📖 Step 4: Generating planting guide...")
        guide = ""
        if recommended_plants:
            assistant = GardeningAssistant(language=language)
            guide = assistant.generate_planting_guide(recommended_plants)
        else:
            guide = "No ideal plants found for the current weather conditions and available space. Consider trying again in a different season or expanding your growing area!"

        # 5. Compile Results
        results = {
            "success": True,
            "space_analysis": space_analysis_results,
            "overlay_image_url": overlay_path,
            "weather_forecast": forecast_df.to_dict(orient='records'),
            "forecast_summary": forecast_summary,
            "recommended_plants": recommended_plants,
            "planting_guide": guide,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"  - Found {len(recommended_plants)} suitable plants")
        print(f"  - Generated guide in {language}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in full analysis: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


# --- Optional: Add a simple test function ---
def test_components():
    """Test individual components for debugging"""
    print("🧪 Testing individual components...")
    
    # Test Weather Forecaster
    print("\n1. Testing Weather Forecaster...")
    forecaster = WeatherForecaster()
    df, summary = forecaster.get_weather_forecast()
    print(f"Weather forecast rows: {len(df)}")
    
    # Test Recommendation Engine
    print("\n2. Testing Recommendation Engine...")
    engine = RecommendationEngine()
    print(f"Plant database size: {len(engine.plant_db) if not engine.plant_db.empty else 0}")
    
    # Test Gardening Assistant
    print("\n3. Testing Gardening Assistant...")
    assistant = GardeningAssistant('english')
    print(f"Assistant initialized: {assistant.client is not None}")
    
    print("\n✅ Component testing completed!")


if __name__ == "__main__":
    # Uncomment to run component tests
    # test_components()
    pass