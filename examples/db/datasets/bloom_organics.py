"""
Bloom Organics dataset example for golden-dataset.
"""

import datetime
import os
import sys

# Add parent directory to path to allow importing from examples
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from golden_dataset import GoldenSession, golden

from ..models import Brand, Brandkit, Event, Font


@golden(dependencies=["base"], title="My title", description="My description")
def bloom_organics(session: GoldenSession, base):
    """
    Bloom Organics dataset with a brand and related data.

    Args:
        session: Session to add objects to.
    """
    # Create a brand
    brand = Brand(
        code="bloom_organics",
        name="Bloom Organics",
        description="Clean, plant-based skincare formulated with certified organic ingredients. Simple, effective products for all skin types.",
        logo_url="https://example.com/logos/bloom-organics.png",
        website_url="https://bloomorganics.example.com",
        category="Beauty & Skincare",
        brand_tone="Natural, transparent, gentle",
        core_customer="Health-conscious consumers, 25-55, interested in clean beauty and sustainability",
        synced_at=datetime.datetime.fromisoformat("2025-02-22T10:10:00+00:00"),
        created_at=datetime.datetime.fromisoformat("2024-06-25T08:35:00+00:00"),
        updated_at=datetime.datetime.fromisoformat("2025-02-22T10:10:00+00:00"),
    )
    session.add(brand)
    session.refresh(brand)

    # Create fonts
    font1 = Font(
        brand_id=brand.id,
        font_family="Cormorant Garamond",
        font_weight=500,
        font_style="normal",
        source="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500&display=swap",
    )
    session.add(font1)
    session.refresh(font1)

    font2 = Font(
        brand_id=brand.id,
        font_family="Cormorant Garamond",
        font_weight=300,
        font_style="italic",
        source="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,300&display=swap",
    )
    session.add(font2)
    session.refresh(font2)

    # Create brandkits
    brandkit1 = Brandkit(
        brand_id=brand.id,
        name="Bloom Organics - Natural Beauty",
        voice_id="XB0fDUnXU5powFXDhCwa",
        primary_text_color="#445C3C",
        primary_background_color="#F8F9F3",
        secondary_text_color="#E2C4A6",
        secondary_background_color="#FFFFFF",
        primary_font_id=font1.id,
        secondary_font_id=font2.id,
    )
    session.add(brandkit1)

    brandkit2 = Brandkit(
        brand_id=brand.id,
        name="Bloom Organics - Minimalist Edition",
        primary_text_color="#445C3C",
        primary_background_color="#FFFFFF",
        primary_font_id=font1.id,
    )
    session.add(brandkit2)

    # Create events
    event1 = Event(
        brand_id=brand.id,
        name="Skincare Product Purchase",
        event_type_id=0,  # Purchase
        value=1.0,
        active=True,
        synced_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
        created_at=datetime.datetime.fromisoformat("2024-10-25T12:16:00+00:00"),
        updated_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
    )
    session.add(event1)

    event2 = Event(
        brand_id=brand.id,
        name="Skincare Product Page View",
        event_type_id=1,  # Landing page
        value=0.3,
        active=True,
        synced_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
        created_at=datetime.datetime.fromisoformat("2024-10-25T12:17:00+00:00"),
        updated_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
    )
    session.add(event2)

    event3 = Event(
        brand_id=brand.id,
        name="Clean Beauty Email Signup",
        event_type_id=2,  # Sign-Up
        value=0.4,
        active=True,
        synced_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
        created_at=datetime.datetime.fromisoformat("2024-10-25T12:25:00+00:00"),
        updated_at=datetime.datetime.fromisoformat("2025-03-03T15:55:00+00:00"),
    )
    session.add(event3)
