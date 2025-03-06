"""
SQLAlchemy models converted from Pydantic models for golden-dataset example.
"""

import datetime
import os
import uuid
from enum import Enum

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "app.db"))
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# Create engine with appropriate settings for SQLite
engine = create_engine(
    DATABASE_URL,
    # Common SQLite-specific engine configuration
    echo=False,  # Set to True to log all SQL statements (useful for debugging)
    connect_args={"check_same_thread": False},  # Allow access from multiple threads
)

# Create session factory
# This is what you'll import elsewhere in your application
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a thread-local session for web applications
# This is useful for web frameworks where you might have multiple
# requests handled concurrently in different threads
Session = scoped_session(SessionLocal)

Base = declarative_base()


def generate_uuid_str():
    """Generate a string representation of a UUID."""
    return str(uuid.uuid4())


class FontStyle(str, Enum):
    NORMAL = "normal"
    ITALIC = "italic"


class EventType(Base):
    __tablename__ = "event_types"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)


class Brand(Base):
    __tablename__ = "brands"

    id = Column(String, primary_key=True, default=generate_uuid_str)
    code = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    website_url = Column(String, nullable=True)
    category = Column(String, nullable=True)
    brand_tone = Column(String, nullable=True)
    core_customer = Column(String, nullable=True)
    synced_at = Column(DATETIME, nullable=True)
    created_at = Column(DATETIME, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(DATETIME, nullable=True, onupdate=datetime.datetime.utcnow)


class Font(Base):
    __tablename__ = "fonts"

    id = Column(String, primary_key=True, default=generate_uuid_str)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    font_family = Column(String, nullable=False)
    font_weight = Column(Integer, nullable=False)
    font_style = Column(SQLAEnum(FontStyle), nullable=False)
    source = Column(String, nullable=False)


class Brandkit(Base):
    __tablename__ = "brandkits"

    id = Column(String, primary_key=True, default=generate_uuid_str)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    name = Column(String, nullable=False)
    voice_id = Column(String, nullable=True)
    primary_text_color = Column(String, nullable=True)
    primary_background_color = Column(String, nullable=True)
    secondary_text_color = Column(String, nullable=True)
    secondary_background_color = Column(String, nullable=True)
    primary_font_id = Column(String, ForeignKey("fonts.id"), nullable=True)
    secondary_font_id = Column(String, ForeignKey("fonts.id"), nullable=True)


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, default=generate_uuid_str)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    name = Column(String, nullable=False)
    event_type_id = Column(Integer, ForeignKey("event_types.id"), nullable=False)
    value = Column(Float, nullable=False, default=1.0)
    active = Column(Boolean, nullable=False, default=False)
    synced_at = Column(DATETIME, nullable=True)
    created_at = Column(DATETIME, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(DATETIME, nullable=True, onupdate=datetime.datetime.utcnow)
