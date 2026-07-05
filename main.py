#!/usr/bin/env python3
"""
2050 Sri Lanka - Facebook Auto-Post Bot
Powered by Hugging Face SDXL (image) + Gemini 2.5 Flash (text)
Author: HackerAI
"""

import os
import json
import time
import random
import logging
import base64
import io
import urllib.parse
import requests
import http.server
import socketserver
import threading
import schedule  # <-- FIXED: Added missing import
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image

# ============================================================
# 1. LIBRARY INSTALLATION (Run once)
# ============================================================
"""
pip install --upgrade google-genai requests schedule python-dotenv Pillow
"""

# ============================================================
# 2. CONFIGURATION
# ============================================================

# --- API Keys (Set as environment variables or use .env file) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "YOUR_HUGGINGFACE_TOKEN_HERE")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "YOUR_FACEBOOK_PAGE_TOKEN_HERE")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "YOUR_FACEBOOK_PAGE_ID_HERE")

# --- Paths ---
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "generated_images"
POSTED_LOG = BASE_DIR / "posted_log.json"
SCHEDULE_LOG = BASE_DIR / "schedule_log.json"

# --- Ensure directories exist ---
IMAGES_DIR.mkdir(exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 3. GEMINI API INTEGRATION (FIXED - using Hugging Face SDXL)
# ============================================================

from google import genai as google_genai
from google.genai import types

class GeminiGenerator:
    """Handles all API interactions"""

    def __init__(self, api_key: str):
        self.client = google_genai.Client(api_key=api_key)

    def generate_image_via_huggingface(self, prompt: str) -> Optional[bytes]:
        """
        Generate image using Hugging Face SDXL (free, high quality).
        Requires HUGGINGFACE_TOKEN environment variable.
        """
        try:
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
            
            logger.info(f"🎨 Generating image via Hugging Face SDXL: {prompt[:50]}...")
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text",
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                        "width": 1024,
                        "height": 1024
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200 and len(response.content) > 1000:
                logger.info(f"✅ Image generated: {len(response.content)} bytes")
                return response.content
            elif response.status_code == 503:
                logger.warning("⚠️ Hugging Face model is loading. Retrying in 20s...")
                time.sleep(20)
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text",
                            "num_inference_steps": 30,
                            "guidance_scale": 7.5,
                            "width": 1024,
                            "height": 1024
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200 and len(response.content) > 1000:
                    logger.info(f"✅ Image generated on retry: {len(response.content)} bytes")
                    return response.content
                else:
                    logger.error(f"❌ Hugging Face retry failed: {response.status_code}")
                    return None
            else:
                logger.error(f"❌ Hugging Face error: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Hugging Face exception: {e}")
            return None

    def generate_category_image(self, category: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Generate a highly realistic, possible future image of Sri Lanka in 2050.
        Uses Hugging Face SDXL for image and Gemini for captions.
        Returns: (image_bytes, caption_with_hashtags)
        """
        image_prompt = f"""
        Photorealistic ultra-high quality image of Sri Lanka in 2050.
        Category: {category}
        Ultra realistic photography style, must look like a REAL photograph,
        8K resolution quality, sharp details, cinematic lighting with natural golden hour,
        clear blue sky, fluffy white clouds, lush tropical vegetation,
        modern sustainable infrastructure, award-winning travel photography.
        """

        caption_prompt = f"""
        Write a SINGLE engaging Facebook post caption in Sinhala language for an image
        showing '{category}' in Sri Lanka in 2050.

        Requirements:
        - Write ONLY the caption text, no explanations
        - 3-5 sentences describing the image and its future vision
        - Inspiring and positive tone
        - At the end, include 8-10 relevant hashtags in both Sinhala and English
        - Hashtags should include: #SriLanka2050 #FutureSriLanka and relevant others
        - Make it emotional and thought-provoking for Sri Lankan audience
        - DO NOT use any markdown formatting like ** or ##
        - Use simple, flowing Sinhala language
        """

        try:
            image_bytes = self.generate_image_via_huggingface(image_prompt)

            if not image_bytes:
                logger.error(f"No image generated for category: {category}")
                return None, None

            logger.info(f"📝 Generating caption for category: {category}")
            caption = None
            
            for attempt in range(3):
                try:
                    cap_response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=caption_prompt
                    )
                    caption = cap_response.text.strip() if cap_response.text else None
                    if caption:
                        break
                except Exception as cap_e:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"⚠️ Caption attempt {attempt+1} failed: {cap_e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            if not caption:
                caption = f"2050 දී ශ්‍රී ලංකාවේ {category} #SriLanka2050 #FutureSriLanka"

            return image_bytes, caption

        except Exception as e:
            logger.error(f"Generation error for {category}: {e}")
            return None, None

    def generate_historical_future_images(self, place: str, place_en: str) -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes], Optional[str]]:
        """
        Generate 3 images for a famous Sri Lankan place using Hugging Face SDXL.
        1. Historically accurate past view
        2. Future possible view (2050)  
        3. Environmental destruction view
        Returns: (past_img, future_img, destruction_img, caption)
        """
        
        past_prompt = f"""
        Historically accurate photograph of {place} ({place_en}), Sri Lanka
        in its original golden age. National Geographic documentary style,
        historically accurate architecture and structures, natural lighting,
        8K cinematic quality, golden hour warm sunlight creating dramatic shadows,
        vintage film quality, lush greenery, photorealistic.
        """

        future_prompt = f"""
        Photorealistic image of {place} ({place_en}), Sri Lanka in 2050
        with sustainable development and preservation. Heritage site preserved,
        solar-powered facilities discreetly placed, electric autonomous shuttles,
        smart preservation systems, reforested surrounding areas, green technology,
        clear blue sky with fluffy white clouds, cinematic golden hour lighting,
        8K ultra realistic photography, harmonious sustainable future.
        """

        destruction_prompt = f"""
        Photorealistic devastating environmental destruction of {place} ({place_en}), 
        Sri Lanka. Plastic waste piles, garbage, graffiti on ancient walls,
        dry cracked mud, toxic green algae, dead brown trees, polluted grey sky,
        muted desaturated colors, photojournalism style, Pulitzer Prize winning photo,
        wide angle lens showing full scale of destruction, ultra realistic 8K.
        """

        caption_prompt = f"""
        Write a Facebook post caption in Sinhala language about {place} ({place_en}) in Sri Lanka.

        The post includes 3 images showing:
        1. The place in its original/historical glory
        2. How it could look in 2050 with proper sustainable development
        3. Environmental destruction if humans remain negligent

        Requirements:
        - Write ONLY the caption text, no introductions or explanations
        - 5-7 sentences in flowing, emotional Sinhala
        - Explain the dramatic contrast between the three images
        - Raise awareness about protecting Sri Lanka's cultural heritage
        - Strong call to action for heritage preservation
        - Emotional, inspiring, educational tone
        - At the end, include 8-10 hashtags
        - Include: #HeritageProtection #{place_en.replace(' ', '')} #SriLanka #FutureVsDestruction
        - DO NOT use markdown formatting like ** or ##
        - Make it shareable, emotional, and thought-provoking
        """

        def _generate_single_image(prompt: str, label: str) -> Optional[bytes]:
            try:
                logger.info(f"[{label}] Generating...")
                return self.generate_image_via_huggingface(prompt)
            except Exception as e:
                logger.error(f"❌ {label} error: {e}")
                return None

        try:
            logger.info(f"🏛️ Generating 3 images for: {place} ({place_en})")

            past_bytes = _generate_single_image(past_prompt, "1/3 Past")
            future_bytes = _generate_single_image(future_prompt, "2/3 Future 2050")
            destruction_bytes = _generate_single_image(destruction_prompt, "3/3 Destruction")

            images = [img for img in [past_bytes, future_bytes, destruction_bytes] if img is not None]
            if len(images) < 2:
                logger.error(f"❌ Not enough images generated for historical: {place} ({place_en})")
                return past_bytes, future_bytes, destruction_bytes, None

            logger.info(f"📝 Generating Sinhala caption...")
            caption = None
            
            for attempt in range(3):
                try:
                    cap_response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=caption_prompt
                    )
                    caption = cap_response.text.strip() if cap_response.text else None
                    if caption:
                        break
                except Exception as cap_e:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"⚠️ Caption attempt {attempt+1} failed: {cap_e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            if not caption:
                caption = f"{place} - අතීතය, 2050 අනාගතය, සහ විනාශය #HeritageProtection #SriLanka"

            return past_bytes, future_bytes, destruction_bytes, caption

        except Exception as e:
            logger.error(f"Generation error for {place}: {e}")
            return None, None, None, None


# ============================================================
# 4. FACEBOOK GRAPH API INTEGRATION
# ============================================================

class FacebookPublisher:
    """Handles all Facebook Page posting via Graph API"""

    def __init__(self, page_id: str, access_token: str):
        self.page_id = page_id
        self.access_token = access_token
        self.api_base = "https://graph.facebook.com/v19.0"

    def post_single_image(self, image_bytes: bytes, caption: str) -> bool:
        try:
            url = f"{self.api_base}/{self.page_id}/photos"
            
            temp_path = IMAGES_DIR / f"temp_post_{int(time.time())}.jpg"
            
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img = img.convert("RGB")
                img.save(temp_path, "JPEG", quality=95)
            except:
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)

            with open(temp_path, "rb") as image_file:
                response = requests.post(
                    url,
                    params={"access_token": self.access_token},
                    files={"source": image_file},
                    data={
                        "caption": caption,
                        "published": "true"
                    }
                )

            temp_path.unlink(missing_ok=True)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Image posted successfully! Post ID: {result.get('id', 'N/A')}")
                return True
            else:
                logger.error(f"❌ Facebook API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Facebook post error: {e}")
            return False

    def post_multi_image(self, image_bytes_list: List[bytes], caption: str) -> bool:
        try:
            if not image_bytes_list:
                logger.error("No images to post")
                return False

            media_ids = []
            temp_files = []

            for i, img_bytes in enumerate(image_bytes_list):
                if not img_bytes:
                    continue

                temp_path = IMAGES_DIR / f"temp_multi_{int(time.time())}_{i}.jpg"
                
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.convert("RGB")
                    img.save(temp_path, "JPEG", quality=95)
                except:
                    with open(temp_path, "wb") as f:
                        f.write(img_bytes)
                
                temp_files.append(temp_path)

                upload_url = f"{self.api_base}/{self.page_id}/photos"
                with open(temp_path, "rb") as img_file:
                    upload_resp = requests.post(
                        upload_url,
                        params={
                            "access_token": self.access_token,
                            "published": "false"
                        },
                        files={"source": img_file}
                    )

                if upload_resp.status_code == 200:
                    media_id = upload_resp.json().get("id")
                    if media_id:
                        media_ids.append(media_id)
                        logger.info(f"✅ Uploaded image {i+1}, media ID: {media_id}")
                else:
                    logger.error(f"Upload failed for image {i+1}: {upload_resp.text}")

            for tp in temp_files:
                tp.unlink(missing_ok=True)

            if not media_ids:
                logger.error("No media IDs obtained")
                return False

            post_url = f"{self.api_base}/{self.page_id}/feed"
            
            media_data = [{"media_fbid": mid} for mid in media_ids]

            post_data = {
                "access_token": self.access_token,
                "message": caption,
                "attached_media": json.dumps(media_data),
                "published": "true"
            }

            response = requests.post(post_url, data=post_data)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Multi-image post published! Post ID: {result.get('id', 'N/A')}")
                return True
            else:
                logger.error(f"❌ Post creation error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Multi-image Facebook post error: {e}")
            return False


# ============================================================
# 5. SCHEDULING & POST MANAGEMENT
# ============================================================

class PostScheduler:
    """Manages the daily schedule and prevents duplicate posts"""

    def __init__(self):
        self.log_file = POSTED_LOG
        self.schedule_file = SCHEDULE_LOG
        self.posted_log = self._load_json(self.log_file, {})
        self.schedule_data = self._load_json(self.schedule_file, {})

    def _load_json(self, filepath: Path, default):
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
        return default

    def _save_json(self, data, filepath: Path):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Could not save {filepath}: {e}")

    def mark_posted(self, post_type: str, identifier: str):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.posted_log:
            self.posted_log[today] = {"category": [], "historical": []}
        
        if post_type == "category":
            if identifier not in self.posted_log[today]["category"]:
                self.posted_log[today]["category"].append(identifier)
        elif post_type == "historical":
            if identifier not in self.posted_log[today]["historical"]:
                self.posted_log[today]["historical"].append(identifier)
        
        self._save_json(self.posted_log, self.log_file)

    def is_posted_today(self, post_type: str, identifier: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.posted_log:
            return False
        
        if post_type == "category":
            return identifier in self.posted_log[today].get("category", [])
        elif post_type == "historical":
            return identifier in self.posted_log[today].get("historical", [])
        return False

    def get_daily_category_schedule(self) -> List[str]:
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in self.schedule_data and "category_times" in self.schedule_data[today]:
            return self.schedule_data[today]["category_times"]
        
        sl_times = [
            "07:30", "08:45", "10:00", "11:15", "13:30",
            "14:00", "15:30", "17:00", "18:45", "22:57"
        ]
        
        if today not in self.schedule_data:
            self.schedule_data[today] = {}
        self.schedule_data[today]["category_times"] = sl_times
        self._save_json(self.schedule_data, self.schedule_file)
        
        return sl_times

    def get_daily_historical_schedule(self) -> List[str]:
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in self.schedule_data and "historical_times" in self.schedule_data[today]:
            return self.schedule_data[today]["historical_times"]
        
        sl_times = ["22:58", "22:59", "23:01"]
        
        if today not in self.schedule_data:
            self.schedule_data[today] = {}
        self.schedule_data[today]["historical_times"] = sl_times
        self._save_json(self.schedule_data, self.schedule_file)
        
        return sl_times

    def get_pending_summary(self, categories: list, historical_places: list) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        
        cat_times = self.get_daily_category_schedule()
        hist_times = self.get_daily_historical_schedule()
        
        summary_parts = []
        summary_parts.append(f"\n📅 අද ({today}) වේලාසටහන (ශ්‍රී ලංකා වේලාව - SLST):")
        summary_parts.append(f"{'='*50}")
        
        summary_parts.append(f"\n📸 Category Posts (10):")
        for i, time_str in enumerate(cat_times):
            scheduled_time = datetime.strptime(time_str, "%H:%M").time()
            cat_name = categories[i] if i < len(categories) else f"Category {i+1}"
            posted = self.is_posted_today("category", categories[i]) if i < len(categories) else False
            
            if scheduled_time > now.time() and not posted:
                status = "⏳ බලා සිටී"
            elif posted:
                status = "✅ පළ කළා"
            else:
                status = "⏳ බලා සිටී"
            
            summary_parts.append(f"   {time_str} - {cat_name} {status}")
        
        summary_parts.append(f"\n🏛️ Historical Posts (3 වතාවක්):")
        for time_str in hist_times:
            scheduled_time = datetime.strptime(time_str, "%H:%M").time()
            
            any_posted = False
            for place_si, place_en in historical_places:
                if self.is_posted_today("historical", f"{place_si} ({place_en})"):
                    any_posted = True
                    break
            
            if scheduled_time > now.time() and not any_posted:
                status = "⏳ බලා සිටී"
            elif any_posted:
                status = "✅ පළ කළා"
            else:
                status = "⏳ බලා සිටී"
            
            summary_parts.append(f"   {time_str} - Historical Place Post {status}")
        
        summary_parts.append(f"\n{'='*50}")
        
        return "\n".join(summary_parts)


# ============================================================
# 6. MAIN BOT ENGINE
# ============================================================

class SriLanka2050Bot:
    """Main bot orchestrating all components"""

    def __init__(self):
        self.gemini = GeminiGenerator(GEMINI_API_KEY)
        self.facebook = FacebookPublisher(FACEBOOK_PAGE_ID, FACEBOOK_PAGE_ACCESS_TOKEN)
        self.scheduler = PostScheduler()
        
        self.categories = [
            "ආර්ථිකය (Economy)",
            "අධ්‍යාපනය (Education)",
            "තාක්ෂණික දියුණුව (Technology)",
            "හරිත නගර (Green Cities)",
            "පුනර්ජනනීය බලශක්තිය (Renewable Energy)",
            "සංස්කෘතිය (Culture)",
            "ප්‍රවාහනය (Transportation)",
            "කෘෂිකර්මය (Agriculture)",
            "සෞඛ්‍ය සේවා (Healthcare)",
            "සංචාරක කර්මාන්තය (Tourism)"
        ]
        
        self.historical_places = [
            ("සීගිරිය", "Sigiriya"),
            ("ගාල්ල කොටුව", "Galle Fort"),
            ("ශ්‍රී මහා බෝධිය", "Sri Maha Bodhiya"),
            ("යාපනය බලකොටුව", "Jaffna Fort"),
            ("මිහින්තලය", "Mihintale"),
            ("දඹුල්ල රජ මහා විහාරය", "Dambulla Cave Temple"),
            ("කැළණිය රජ මහා විහාරය", "Kelaniya Temple"),
            ("පොළොන්නරුව", "Polonnaruwa"),
            ("කොළඹ වරාය", "Colombo Port"),
            ("නුවරඑළිය", "Nuwara Eliya")
        ]

    def run_category_post(self, category_index: int):
        if category_index < 0 or category_index >= len(self.categories):
            logger.error(f"Invalid category index: {category_index}")
            return False

        category = self.categories[category_index]
        
        if self.scheduler.is_posted_today("category", category):
            logger.info(f"⏭️ Already posted category '{category}' today. Skipping.")
            return False

        logger.info(f"🎯 Generating category post: {category}")
        
        image_bytes, caption = self.gemini.generate_category_image(category)
        
        if not image_bytes:
            logger.error(f"❌ Failed to generate image for {category}")
            return False
        
        if not caption:
            caption = f"2050 දී ශ්‍රී ලංකාවේ {category} #SriLanka2050 #FutureSriLanka"
        
        success = self.facebook.post_single_image(image_bytes, caption)
        
        if success:
            self.scheduler.mark_posted("category", category)
            logger.info(f"✅ Successfully posted category: {category}")
        
        return success

    def run_historical_post(self, place_index: int):
        if place_index < 0 or place_index >= len(self.historical_places):
            logger.error(f"Invalid place index: {place_index}")
            return False

        place_si, place_en = self.historical_places[place_index]
        identifier = f"{place_si} ({place_en})"
        
        if self.scheduler.is_posted_today("historical", identifier):
            logger.info(f"⏭️ Already posted historical '{identifier}' today. Skipping.")
            return False

        logger.info(f"🏛️ Generating historical post: {identifier}")
        
        past_img, future_img, destruction_img, caption = self.gemini.generate_historical_future_images(place_si, place_en)
        
        images = [img for img in [past_img, future_img, destruction_img] if img is not None]
        
        if len(images) < 2:
            logger.error(f"❌ Not enough images generated for historical: {identifier}")
            
            for i, img in enumerate([past_img, future_img, destruction_img]):
                if img:
                    debug_path = IMAGES_DIR / f"debug_{place_en}_{i}.png"
                    with open(debug_path, "wb") as f:
                        f.write(img)
                    logger.info(f"Saved debug image: {debug_path}")
            
            return False
        
        if not caption:
            caption = f"{place_si} - අතීතය, 2050 අනාගතය, සහ විනාශය #HeritageProtection #SriLanka"
        
        success = self.facebook.post_multi_image(images, caption)
        
        if success:
            self.scheduler.mark_posted("historical", identifier)
            logger.info(f"✅ Successfully posted historical: {identifier}")
        
        return success

    def run_full_category_cycle(self):
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"===== Starting Category Post Cycle: {today} =====")
        
        for i, category in enumerate(self.categories):
            if not self.scheduler.is_posted_today("category", category):
                logger.info(f"Posting category {i+1}/10: {category}")
                self.run_category_post(i)
                time.sleep(random.randint(30, 90))
            else:
                logger.info(f"Category already posted today: {category}")
        
        logger.info(f"===== Category Post Cycle Complete: {today} =====")

    def run_full_historical_cycle(self):
        today = datetime.now().strftime("%Y-%m-%d")
        
        for i, (place_si, place_en) in enumerate(self.historical_places):
            identifier = f"{place_si} ({place_en})"
            if not self.scheduler.is_posted_today("historical", identifier):
                logger.info(f"🏛️ Posting historical place: {identifier}")
                self.run_historical_post(i)
                return
        
        logger.info("✅ All historical places posted for today!")


# ============================================================
# 7. SCHEDULE SETUP & MAIN LOOP
# ============================================================

def setup_schedules(bot: SriLanka2050Bot):
    """Set up the daily schedule."""
    
    category_times = bot.scheduler.get_daily_category_schedule()
    for i, time_str in enumerate(category_times):
        schedule.every().day.at(time_str).do(
            lambda idx=i: scheduled_category_post(bot, idx)
        )
    
    historical_times = bot.scheduler.get_daily_historical_schedule()
    for time_str in historical_times:
        schedule.every().day.at(time_str).do(
            lambda: scheduled_historical_post(bot)
        )


def scheduled_category_post(bot: SriLanka2050Bot, index: int):
    if index < len(bot.categories):
        category = bot.categories[index]
        if not bot.scheduler.is_posted_today("category", category):
            logger.info(f"⏰ Scheduled time reached for category: {category}")
            bot.run_category_post(index)
        else:
            logger.info(f"⏭️ Category already posted today: {category}")
    return schedule.CancelJob


def scheduled_historical_post(bot: SriLanka2050Bot):
    for i, (place_si, place_en) in enumerate(bot.historical_places):
        identifier = f"{place_si} ({place_en})"
        if not bot.scheduler.is_posted_today("historical", identifier):
            logger.info(f"⏰ Scheduled time reached for historical: {identifier}")
            bot.run_historical_post(i)
            return schedule.CancelJob
    
    logger.info("✅ All historical places posted for today!")
    return schedule.CancelJob


# ============================================================
# 8. RAILWAY HEALTH CHECK SERVER
# ============================================================

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"2050 Sri Lanka Bot is running")
    
    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.getenv("PORT", 8080))
    handler = HealthCheckHandler
    httpd = socketserver.TCPServer(("", port), handler)
    logger.info(f"✅ Health check server running on port {port}")
    httpd.serve_forever()


# ============================================================
# 9. ENTRY POINT
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("🚀 2050 Sri Lanka Facebook Auto-Post Bot Starting...")
    logger.info("=" * 60)
    
    if "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY:
        logger.error("❌ Please set your GEMINI_API_KEY environment variable!")
        return
    
    if "YOUR_HUGGINGFACE_TOKEN_HERE" in HUGGINGFACE_TOKEN:
        logger.error("❌ Please set your HUGGINGFACE_TOKEN environment variable!")
        logger.error("   Get it from: https://huggingface.co/settings/tokens")
        return
    
    if "YOUR_FACEBOOK_PAGE_TOKEN_HERE" in FACEBOOK_PAGE_ACCESS_TOKEN:
        logger.error("❌ Please set your FACEBOOK_PAGE_ACCESS_TOKEN!")
        return
    
    if "YOUR_FACEBOOK_PAGE_ID_HERE" in FACEBOOK_PAGE_ID:
        logger.error("❌ Please set your FACEBOOK_PAGE_ID!")
        return
    
    bot = SriLanka2050Bot()
    setup_schedules(bot)
    
    logger.info("""
    📋 BOT SCHEDULE SUMMARY:
    ========================
    📸 Category Posts: 10 per day
    
    🏛️ Historical Place Posts: 1 place per day × 3 times
    
    🎨 Image Model: Hugging Face SDXL (high quality, free)
    💬 Text Model: Gemini 2.5 Flash (with retry logic)
    """)
    
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    logger.info(f"🩺 Health check server started")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        raise


# ============================================================
# 10. MANUAL RUN OPTIONS (For testing)
# ============================================================

def manual_test_category(bot: SriLanka2050Bot, index: int = 0):
    logger.info(f"🧪 Manual test: Category {index}")
    bot.run_category_post(index)

def manual_test_historical(bot: SriLanka2050Bot, index: int = 0):
    logger.info(f"🧪 Manual test: Historical {index}")
    bot.run_historical_post(index)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        bot = SriLanka2050Bot()
        
        if sys.argv[1] == "--test-category":
            index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            manual_test_category(bot, index)
        elif sys.argv[1] == "--test-historical":
            index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            manual_test_historical(bot, index)
        elif sys.argv[1] == "--full-category":
            bot.run_full_category_cycle()
        elif sys.argv[1] == "--full-historical":
            bot.run_full_historical_cycle()
        elif sys.argv[1] == "--schedule":
            bot = SriLanka2050Bot()
            print(bot.scheduler.get_pending_summary(bot.categories, bot.historical_places))
        else:
            print("Usage:")
            print("  python bot.py                    # Run scheduled bot")
            print("  python bot.py --test-category 0  # Test category 0")
            print("  python bot.py --test-historical 0 # Test historical 0")
            print("  python bot.py --full-category    # Post all categories now")
            print("  python bot.py --full-historical  # Post next historical now")
            print("  python bot.py --schedule         # Show today's schedule")
    else:
        main()
