#!/usr/bin/env python3
"""
2050 Sri Lanka - Facebook Auto-Post Bot
Powered by Gemini 2.5 Flash (text + image from pollinations.ai)
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
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image

# ============================================================
# 1. LIBRARY INSTALLATION
# ============================================================
"""
pip install --upgrade google-genai requests schedule python-dotenv Pillow
"""

# ============================================================
# 2. CONFIGURATION
# ============================================================

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "YOUR_FACEBOOK_PAGE_TOKEN_HERE")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "YOUR_FACEBOOK_PAGE_ID_HERE")

# --- Paths ---
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "generated_images"
POSTED_LOG = BASE_DIR / "posted_log.json"
SCHEDULE_LOG = BASE_DIR / "schedule_log.json"

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
# 3. POLLINATIONS.AI IMAGE GENERATION (REPLACED GEMINI IMAGE)
# ============================================================

from google import genai as google_genai
from google.genai import types

class GeminiGenerator:
    """Handles all API interactions - images via pollinations.ai, text via Gemini"""

    def __init__(self, api_key: str):
        self.client = google_genai.Client(api_key=api_key)

    def generate_image_via_pollinations(self, prompt: str, retry_on_daily_quota: bool = False, schedule_time: str = None, schedule_key: str = None) -> Optional[bytes]:
        """
        Generate high-quality image using pollinations.ai (free, no API key needed).
        Uses maximum quality settings: 8K resolution, enhanced prompt.
        """
        max_retries = 5
        base_delay = 10
        
        # Enhance the prompt for maximum quality without changing original meaning
        # We preserve the original prompt but add quality-focused suffixes
        quality_suffix = " --ar 16:9 --v 6.1 --style raw --stylize 1000 --quality 2 --no text watermark signature"
        
        # pollinations.ai uses seed for consistency and model selection
        encoded_prompt = urllib.parse.quote(prompt + quality_suffix)
        
        # Multiple resolution options - we'll try the highest first
        # pollinations.ai supports: width x height, we use 16:9 aspect ratio
        width = 1920
        height = 1080
        
        # pollinations.ai URL format: https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&seed={s}&nologo=true&model={model}
        urls_to_try = [
            # Primary: ultra-HD with best model
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={random.randint(1, 999999)}&nologo=true&model=flux-pro&enhance=true&private=true",
            # Fallback: without enhance
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={random.randint(1, 999999)}&nologo=true&model=flux-pro&private=true",
            # Second fallback: standard model but high resolution
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={random.randint(1, 999999)}&nologo=true&model=flux&private=true",
            # Third fallback: 4K resolution
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=3840&height=2160&seed={random.randint(1, 999999)}&nologo=true&model=flux&private=true",
            # Final fallback: default model
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={random.randint(1, 999999)}&nologo=true",
        ]
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🎨 Generating image via pollinations.ai... (attempt {attempt+1}/{max_retries})")
                
                for url_idx, url in enumerate(urls_to_try):
                    try:
                        logger.info(f"📡 Trying pollinations URL variant {url_idx+1}/{len(urls_to_try)}...")
                        response = requests.get(url, timeout=60)
                        
                        if response.status_code == 200 and len(response.content) > 5000:
                            # Verify it's actually an image
                            content_type = response.headers.get('content-type', '')
                            if 'image' in content_type or response.content[:4] in [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\x89PNG', b'GIF8']:
                                logger.info(f"✅ Image generated via pollinations.ai: {len(response.content)} bytes (URL variant {url_idx+1})")
                                return response.content
                            else:
                                logger.warning(f"⚠️ Response not an image (type: {content_type}), trying next variant...")
                        else:
                            logger.warning(f"⚠️ pollinations.ai returned status {response.status_code}, size {len(response.content)} bytes")
                    except requests.exceptions.Timeout:
                        logger.warning(f"⚠️ pollinations.ai timeout on variant {url_idx+1}, trying next...")
                        continue
                    except Exception as url_e:
                        logger.warning(f"⚠️ pollinations.ai error on variant {url_idx+1}: {url_e}")
                        continue
                
                # If we got here, all URL variants failed for this attempt
                if attempt < max_retries - 1:
                    retry_delay = base_delay + (attempt * 15)
                    logger.warning(f"⚠️ All pollinations.ai variants failed. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ All pollinations.ai attempts and variants failed after {max_retries} retries")
                    return None
                    
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    retry_delay = base_delay + (attempt * 10)
                    logger.warning(f"⚠️ pollinations.ai error (attempt {attempt+1}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ pollinations.ai image generation exception after {max_retries} retries: {e}")
                    return None

        return None

    def generate_image_via_gemini(self, prompt: str, retry_on_daily_quota: bool = False, schedule_time: str = None, schedule_key: str = None) -> Optional[bytes]:
        """
        Generate high-quality image using pollinations.ai (replacing Gemini image generation).
        No additional API key needed for pollinations.
        """
        # Delegate to pollinations.ai for image generation
        logger.info("🎨 Using pollinations.ai for image generation (replacing Gemini image)...")
        return self.generate_image_via_pollinations(prompt, retry_on_daily_quota, schedule_time, schedule_key)

    def _retry_scheduled_image(self, prompt: str, schedule_key: str):
        """Retry image generation from a scheduled task (next day)."""
        logger.info(f"🔄 Retrying scheduled image generation for: {schedule_key}")
        return self.generate_image_via_pollinations(prompt)

    def generate_new_historical_place(self, used_places: List[str]) -> Tuple[str, str]:
        """
        Asks Gemini for ONE real, famous historical/cultural/natural heritage site in
        Sri Lanka that is NOT in used_places (so the bot never repeats a place, with
        no fixed list - it can keep finding new places for as long as it runs).
        Returns (place_name_sinhala, place_name_english).
        """
        used_list_str = ", ".join(used_places) if used_places else "none yet"
        prompt = f"""
        You help run a Facebook page about Sri Lanka's heritage sites.
        Give me ONE real, well-known historical, cultural, religious, or natural
        heritage site in Sri Lanka that has NOT been used before.

        Sites already used - do NOT repeat any of these: {used_list_str}

        Respond with EXACTLY one line, nothing else, in this exact format:
        SinhalaName|EnglishName

        Example of the exact format expected: සීගිරිය|Sigiriya

        Do not add numbering, explanations, or markdown - just the single line.
        """

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                text = response.text.strip() if response.text else ""
                first_line = text.splitlines()[0].strip() if text else ""

                if "|" in first_line:
                    place_si, place_en = first_line.split("|", 1)
                    place_si = place_si.strip()
                    place_en = place_en.strip()
                    if place_en and place_en not in used_places:
                        logger.info(f"🆕 New historical place picked: {place_si} ({place_en})")
                        return place_si, place_en
                    else:
                        logger.warning(f"⚠️ Gemini repeated an already-used place ({place_en}), retrying...")
                else:
                    logger.warning(f"⚠️ Unexpected format from Gemini: '{text}', retrying...")
            except Exception as e:
                logger.warning(f"⚠️ Place generation attempt {attempt+1} failed: {e}")
            time.sleep(5)

        # Fallback so the bot never gets permanently stuck if Gemini keeps failing/repeating
        fallback_en = f"Sri Lanka Heritage Site {int(time.time())}"
        logger.error(f"❌ Could not get a unique place from Gemini after retries, using fallback: {fallback_en}")
        return "ශ්‍රී ලංකා උරුම ස්ථානය", fallback_en

    def generate_place_visual_description(self, place_si: str, place_en: str) -> str:
        """
        Asks Gemini for a factual visual description of how this REAL place actually
        looks (architecture, colors, materials, layout, surroundings, landmarks).
        This description is injected into every image prompt for this place so the
        generated images accurately match the real place instead of being generic.
        """
        prompt = f"""
        Describe the REAL, factual visual appearance of {place_en} ({place_si}) in Sri Lanka,
        to be used as a reference for AI image generation. Only describe what this exact
        place genuinely looks like in real life.

        Cover in 4-6 concise sentences, plain text, no markdown:
        - Its distinctive architecture, structures, or natural landform
        - Key colors and materials (stone type, brick, paint, vegetation, water, etc.)
        - Layout and surroundings (terrain, water bodies, forest, urban context)
        - Any unique, instantly recognizable visual features that make it identifiable

        Write ONLY the description, nothing else - no intro, no title.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            description = response.text.strip() if response.text else ""
            if description:
                logger.info(f"📋 Got visual description for {place_en} ({place_si})")
                return description
        except Exception as e:
            logger.warning(f"⚠️ Could not get visual description for {place_en}: {e}")
        return ""

    def generate_historical_future_images(self, place: str, place_en: str, visual_description: str = "") -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes], Optional[str]]:
        """Generate 3 images for a historical place using pollinations.ai."""

        reference_line = f"\n        Accurate real-world visual reference for this exact place: {visual_description}\n" if visual_description else ""

        past_prompt = f"""
        Generate a historically accurate photograph of {place} ({place_en}), Sri Lanka
        in its original golden age.{reference_line}
        National Geographic documentary style,
        historically accurate architecture and structures, natural lighting,
        golden hour warm sunlight creating dramatic shadows,
        vintage film quality, lush greenery, photorealistic.
        """

        future_prompt = f"""
        Generate a photorealistic image of {place} ({place_en}), Sri Lanka in 2050
        with sustainable development and preservation.{reference_line}
        Heritage site preserved,
        solar-powered facilities discreetly placed, electric autonomous shuttles,
        smart preservation systems, reforested surrounding areas, green technology,
        clear blue sky with fluffy white clouds, cinematic golden hour lighting,
        8K ultra realistic photography, harmonious sustainable future.
        """

        destruction_prompt = f"""
        Generate a photorealistic devastating environmental destruction of {place} ({place_en}), 
        Sri Lanka.{reference_line}
        Plastic waste piles, garbage, graffiti on ancient walls,
        dry cracked mud, toxic green algae, dead brown trees, polluted grey sky,
        muted desaturated colors, photojournalism style, wide angle lens
        showing full scale of destruction, ultra realistic.
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
                logger.info(f"[{label}] Generating via pollinations.ai...")
                return self.generate_image_via_pollinations(prompt)
            except Exception as e:
                logger.error(f"❌ {label} error: {e}")
                return None

        try:
            logger.info(f"🏛️ Generating 3 images for: {place} ({place_en})")


            past_bytes = _generate_single_image(past_prompt, "1/3 Past")
            time.sleep(60)
            future_bytes = _generate_single_image(future_prompt, "2/3 Future 2050")
            time.sleep(60)
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
                    logger.warning(f"⚠️ Caption attempt {attempt+1} failed. Retrying...")
                    time.sleep(wait_time)

            if not caption:
                caption = f"{place} - අතීතය, 2050 අනාගතය, සහ විනාශය #HeritageProtection #SriLanka"

            return past_bytes, future_bytes, destruction_bytes, caption

        except Exception as e:
            logger.error(f"Generation error for {place}: {e}")
            return None, None, None, None

    def generate_single_historical_image(self, place: str, place_en: str, image_type: str, schedule_time: str = None, visual_description: str = "") -> Tuple[Optional[bytes], Optional[str]]:
        """Generate a single image for a historical place with its own caption."""

        reference_line = f"\n            Accurate real-world visual reference for this exact place: {visual_description}\n" if visual_description else ""

        if image_type == "past":
            image_prompt = f"""
            Generate a historically accurate photograph of {place} ({place_en}), Sri Lanka
            in its original golden age.{reference_line}
            National Geographic documentary style,
            historically accurate architecture and structures, natural lighting,
            golden hour warm sunlight creating dramatic shadows,
            vintage film quality, lush greenery, photorealistic.
            """
            caption_prompt = f"""
            Write a single Facebook post caption in Sinhala language about the historical glory of {place} ({place_en}), Sri Lanka.

            This post shows {place} in its original golden age.

            Requirements:
            - Write ONLY the caption text, no introductions or explanations
            - 4-6 sentences in flowing, emotional Sinhala
            - Describe the historical significance and beauty of this place
            - End with 8-10 hashtags
            - Include: #Heritage #{place_en.replace(' ', '')} #SriLanka #HistoricalGlory
            - DO NOT use markdown formatting like ** or ##
            """
        elif image_type == "future":
            image_prompt = f"""
            Generate a photorealistic image of {place} ({place_en}), Sri Lanka in 2050
            with sustainable development and preservation.{reference_line}
            Heritage site preserved,
            solar-powered facilities discreetly placed, electric autonomous shuttles,
            smart preservation systems, reforested surrounding areas, green technology,
            clear blue sky with fluffy white clouds, cinematic golden hour lighting,
            8K ultra realistic photography, harmonious sustainable future.
            """
            caption_prompt = f"""
            Write a single Facebook post caption in Sinhala language about {place} ({place_en}), Sri Lanka in 2050.

            This post shows how {place} could look in 2050 with proper preservation and sustainable development.

            Requirements:
            - Write ONLY the caption text, no introductions or explanations
            - 4-6 sentences in flowing, emotional Sinhala
            - Describe the beautiful sustainable future vision for this heritage site
            - End with 8-10 hashtags
            - Include: #SriLanka2050 #FutureHeritage #{place_en.replace(' ', '')} #SustainableSriLanka
            - DO NOT use markdown formatting like ** or ##
            """
        else:  # destruction
            image_prompt = f"""
            Generate a photorealistic devastating environmental destruction of {place} ({place_en}), 
            Sri Lanka.{reference_line}
            Plastic waste piles, garbage, graffiti on ancient walls,
            dry cracked mud, toxic green algae, dead brown trees, polluted grey sky,
            muted desaturated colors, photojournalism style, wide angle lens
            showing full scale of destruction, ultra realistic.
            """
            caption_prompt = f"""
            Write a single Facebook post caption in Sinhala language about the destruction of {place} ({place_en}), Sri Lanka.

            This post shows the devastating environmental destruction and neglect of this heritage site.

            Requirements:
            - Write ONLY the caption text, no introductions or explanations
            - 4-6 sentences in flowing, emotional Sinhala
            - Raise awareness about protecting Sri Lanka's cultural heritage
            - Strong call to action
            - End with 8-10 hashtags
            - Include: #ProtectHeritage #{place_en.replace(' ', '')} #SriLanka #ActNow
            - DO NOT use markdown formatting like ** or ##
            """

        try:
            schedule_key = f"{place_en}_{image_type}"
            image_bytes = self.generate_image_via_pollinations(image_prompt, retry_on_daily_quota=True, schedule_time=schedule_time, schedule_key=schedule_key)
            if not image_bytes:
                return None, None

            logger.info(f"📝 Generating {image_type} caption for {place}...")
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
                    logger.warning(f"⚠️ Caption attempt {attempt+1} failed. Retrying...")
                    time.sleep(wait_time)

            if not caption:
                type_labels = {"past": "ඓතිහාසික තේජස", "future": "2050 අනාගතය", "destruction": "පාරිසරික විනාශය"}
                caption = f"{place} - {type_labels.get(image_type, image_type)} #{place_en.replace(' ', '')} #SriLanka"

            return image_bytes, caption

        except Exception as e:
            logger.error(f"Generation error for {place} {image_type}: {e}")
            return None, None


# ============================================================
# 4. FACEBOOK GRAPH API
# ============================================================

class FacebookPublisher:
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
                    data={"caption": caption, "published": "true"}
                )

            temp_path.unlink(missing_ok=True)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Image posted! Post ID: {result.get('id', 'N/A')}")
                return True
            else:
                # FIX: log the actual response body so auth/permission errors are visible
                logger.error(f"❌ Facebook API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Facebook post error: {e}")
            return False

    def post_multi_image(self, image_bytes_list: List[bytes], caption: str) -> bool:
        try:
            if not image_bytes_list:
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
                        params={"access_token": self.access_token, "published": "false"},
                        files={"source": img_file}
                    )

                if upload_resp.status_code == 200:
                    media_id = upload_resp.json().get("id")
                    if media_id:
                        media_ids.append(media_id)
                else:
                    # FIX: log why an individual photo upload failed
                    logger.error(f"❌ Photo upload error: {upload_resp.status_code} - {upload_resp.text}")

            for tp in temp_files:
                tp.unlink(missing_ok=True)

            if not media_ids:
                return False

            post_url = f"{self.api_base}/{self.page_id}/feed"
            media_data = [{"media_fbid": mid} for mid in media_ids]

            response = requests.post(post_url, data={
                "access_token": self.access_token,
                "message": caption,
                "attached_media": json.dumps(media_data),
                "published": "true"
            })

            if response.status_code == 200:
                logger.info(f"✅ Multi-image post published!")
                return True
            else:
                # FIX: log the actual response body so the real cause is visible
                logger.error(f"❌ Post creation error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Multi-image post error: {e}")
            return False


# ============================================================
# 5. SCHEDULING & POST MANAGEMENT
# ============================================================

class PostScheduler:
    def __init__(self):
        self.log_file = POSTED_LOG
        self.schedule_file = SCHEDULE_LOG
        self.posted_log = self._load_json(self.log_file, {})
        self.schedule_data = self._load_json(self.schedule_file, {})

    def _load_json(self, filepath, default):
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            pass
        return default

    def _save_json(self, data, filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except:
            pass

    def mark_posted(self, post_type, identifier):
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

    def is_posted_today(self, post_type, identifier):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.posted_log:
            return False
        if post_type == "category":
            return identifier in self.posted_log[today].get("category", [])
        elif post_type == "historical":
            return identifier in self.posted_log[today].get("historical", [])
        return False

    def get_daily_historical_schedule(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today in self.schedule_data and "historical_times" in self.schedule_data[today]:
            return self.schedule_data[today]["historical_times"]
        # FIX: the original literal had a broken/unterminated string ("12:08 was missing its
        # closing quote), which is a SyntaxError that stops the whole script from starting.
        sl_times = ["12:06", "12:08", "12:10"]
        if today not in self.schedule_data:
            self.schedule_data[today] = {}
        self.schedule_data[today]["historical_times"] = sl_times
        self._save_json(self.schedule_data, self.schedule_file)
        return sl_times

    def get_historical_image_schedule(self):
        """Returns list of (time, image_type) tuples for historical posts."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today in self.schedule_data and "historical_image_times" in self.schedule_data[today]:
            return self.schedule_data[today]["historical_image_times"]
        # Default: past at 08:40, future at 21:40, destruction at 21:42
        image_times = [
            {"time": "13:08", "image_type": "past"},
            {"time": "13:10", "image_type": "future"},
            {"time": "13:12", "image_type": "destruction"}
        ]
        if today not in self.schedule_data:
            self.schedule_data[today] = {}
        self.schedule_data[today]["historical_image_times"] = image_times
        self._save_json(self.schedule_data, self.schedule_file)
        return image_times

    def get_cached_today_place(self):
        """Returns today's already-picked place (si, en, description) if one was picked, else None."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today in self.schedule_data and "historical_place_si" in self.schedule_data[today]:
            return (
                self.schedule_data[today]["historical_place_si"],
                self.schedule_data[today]["historical_place_en"],
                self.schedule_data[today].get("historical_place_description", "")
            )
        return None

    def get_used_place_names(self):
        """Returns the list of English place names that have already been used (ever), so a new one never repeats."""
        return self.schedule_data.get("_used_historical_places", [])

    def save_today_place(self, place_si, place_en, description=""):
        """Saves today's picked place (with its visual description) and permanently records it as used so it's never picked again."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.schedule_data:
            self.schedule_data[today] = {}
        self.schedule_data[today]["historical_place_si"] = place_si
        self.schedule_data[today]["historical_place_en"] = place_en
        self.schedule_data[today]["historical_place_description"] = description

        used = self.schedule_data.get("_used_historical_places", [])
        if place_en not in used:
            used.append(place_en)
        self.schedule_data["_used_historical_places"] = used

        self._save_json(self.schedule_data, self.schedule_file)

    def mark_historical_image_posted(self, place_identifier, image_type):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.posted_log:
            self.posted_log[today] = {"category": [], "historical": [], "historical_images": {}}
        if "historical_images" not in self.posted_log[today]:
            self.posted_log[today]["historical_images"] = {}
        if place_identifier not in self.posted_log[today]["historical_images"]:
            self.posted_log[today]["historical_images"][place_identifier] = []
        if image_type not in self.posted_log[today]["historical_images"][place_identifier]:
            self.posted_log[today]["historical_images"][place_identifier].append(image_type)
        self._save_json(self.posted_log, self.log_file)

    def is_historical_image_posted(self, place_identifier, image_type):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.posted_log:
            return False
        if "historical_images" not in self.posted_log[today]:
            return False
        if place_identifier not in self.posted_log[today]["historical_images"]:
            return False
        return image_type in self.posted_log[today]["historical_images"][place_identifier]


# ============================================================
# 6. MAIN BOT ENGINE
# ============================================================

class SriLanka2050Bot:
    def __init__(self):
        self.gemini = GeminiGenerator(GEMINI_API_KEY)
        self.facebook = FacebookPublisher(FACEBOOK_PAGE_ID, FACEBOOK_PAGE_ACCESS_TOKEN)
        self.scheduler = PostScheduler()

    def get_todays_place(self):
        """
        Returns today's historical place (place_si, place_en, visual_description). If a
        place was already picked today, reuses it (so past/future/destruction stay
        consistent). Otherwise asks Gemini for a brand-new real Sri Lankan heritage site
        that has never been used before, plus a factual visual description of it, and
        remembers both permanently so the place is never repeated again.
        """
        cached = self.scheduler.get_cached_today_place()
        if cached:
            return cached

        used_places = self.scheduler.get_used_place_names()
        place_si, place_en = self.gemini.generate_new_historical_place(used_places)
        description = self.gemini.generate_place_visual_description(place_si, place_en)
        self.scheduler.save_today_place(place_si, place_en, description)
        return place_si, place_en, description

    def run_historical_post(self):
        place_si, place_en, description = self.get_todays_place()
        identifier = f"{place_si} ({place_en})"
        if self.scheduler.is_posted_today("historical", identifier):
            return False
        
        past_img, future_img, destruction_img, caption = self.gemini.generate_historical_future_images(place_si, place_en, description)
        images = [img for img in [past_img, future_img, destruction_img] if img is not None]
        if len(images) < 2:
            return False
        if not caption:
            caption = f"{place_si} - අතීතය, 2050 අනාගතය, සහ විනාශය #HeritageProtection #SriLanka"
        
        success = self.facebook.post_multi_image(images, caption)
        if success:
            self.scheduler.mark_posted("historical", identifier)
        return success

    def run_historical_single_image_post(self, place_si, place_en, image_type, schedule_time=None, description=""):
        """Generate and post a single historical image with its own caption."""
        identifier = f"{place_si} ({place_en})"
        
        if self.scheduler.is_historical_image_posted(identifier, image_type):
            logger.info(f"⏭️ {image_type} image already posted for {identifier}")
            return False
        
        logger.info(f"🏛️ Generating {image_type} image for: {place_si} ({place_en})")
        image_bytes, caption = self.gemini.generate_single_historical_image(place_si, place_en, image_type, schedule_time, description)
        
        if not image_bytes:
            logger.error(f"❌ Failed to generate {image_type} image for {identifier}")
            return False
        
        if not caption:
            type_labels = {"past": "ඓතිහාසික තේජස", "future": "2050 අනාගතය", "destruction": "පාරිසරික විනාශය"}
            caption = f"{place_si} - {type_labels.get(image_type, image_type)} #{place_en.replace(' ', '')} #SriLanka"
        
        logger.info(f"📤 Posting {image_type} image to Facebook...")
        success = self.facebook.post_single_image(image_bytes, caption)
        
        if success:
            self.scheduler.mark_historical_image_posted(identifier, image_type)
            logger.info(f"✅ {image_type} image posted for {identifier}")
        
        return success


# ============================================================
# 7. SCHEDULE SETUP
# ============================================================

def setup_schedules(bot):
    historical_image_times = bot.scheduler.get_historical_image_schedule()
    for entry in historical_image_times:
        time_str = entry["time"]
        image_type = entry["image_type"]
        schedule.every().day.at(time_str).do(lambda it=image_type, ts=time_str: scheduled_historical_single_image_post(bot, it, ts))

def scheduled_historical_single_image_post(bot, image_type, schedule_time):
    """
    Post the image_type (past/future/destruction) for TODAY's chosen historical place.
    The place is picked fresh via Gemini once per day (no fixed list - a brand-new real
    place every day, never repeated) so that all 3 image types on the same day use the
    same place.
    """
    place_si, place_en, description = bot.get_todays_place()
    identifier = f"{place_si} ({place_en})"

    if not bot.scheduler.is_historical_image_posted(identifier, image_type):
        logger.info(f"⏰ Scheduled time reached for historical {image_type}: {identifier}")
        bot.run_historical_single_image_post(place_si, place_en, image_type, schedule_time, description)
    else:
        logger.info(f"⏭️ {image_type} already posted today for {identifier}")
    return None


# ============================================================
# 8. HEALTH CHECK SERVER
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
    httpd = socketserver.TCPServer(("", port), HealthCheckHandler)
    httpd.serve_forever()


# ============================================================
# 9. ENTRY POINT
# ============================================================

def main():
    logger.info("🚀 2050 Sri Lanka Bot Starting...")
    
    if "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY:
        logger.error("❌ Set GEMINI_API_KEY!")
        return
    if "YOUR_FACEBOOK_PAGE_TOKEN_HERE" in FACEBOOK_PAGE_ACCESS_TOKEN:
        logger.error("❌ Set FACEBOOK_PAGE_ACCESS_TOKEN!")
        return
    if "YOUR_FACEBOOK_PAGE_ID_HERE" in FACEBOOK_PAGE_ID:
        logger.error("❌ Set FACEBOOK_PAGE_ID!")
        return
    
    bot = SriLanka2050Bot()
    setup_schedules(bot)
    
    threading.Thread(target=start_health_server, daemon=True).start()
    logger.info(f"🩺 Health check server started on port {os.getenv('PORT', 8080)}")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped")
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        bot = SriLanka2050Bot()
        if sys.argv[1] == "--test-historical":
            bot.run_historical_post()
        elif sys.argv[1] == "--test-historical-single":
            img_type = sys.argv[2] if len(sys.argv) > 2 else "past"
            place_si, place_en, description = bot.get_todays_place()
            bot.run_historical_single_image_post(place_si, place_en, img_type, description=description)
        elif sys.argv[1] == "--full-historical":
            bot.run_historical_post()
        elif sys.argv[1] == "--schedule":
            print("Bot running - check logs")
        else:
            print("Usage: python bot.py [--test-historical|--test-historical-single TYPE|--full-historical]")
    else:
        main()
