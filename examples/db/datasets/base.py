"""
Base dataset example for golden-dataset.
"""

import os
import sys

# Add parent directory to path to allow importing from examples
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_dataset import golden

from ..models import EventType


@golden
def base(session):
    """
    Base dataset with event types.

    Args:
        session: Session to add objects to.
    """
    # Create event types
    event_type1 = EventType(id=0, name="Purchase")
    event_type2 = EventType(id=1, name="Landing page")
    event_type3 = EventType(id=2, name="Sign-Up")
    event_type4 = EventType(id=3, name="Lead")
    event_type5 = EventType(id=99, name="Other")

    # Add to session
    session.add(event_type1)
    session.add(event_type2)
    session.add(event_type3)
    session.add(event_type4)
    session.add(event_type5)

    # No need to commit - the session will handle that
