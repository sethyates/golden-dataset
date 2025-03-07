from golden_dataset import golden


@golden(title="Test Generator", description="A test generator function")
def sample_generator(session):
    """Generate sample data."""

    # This is just a stub for testing
    class User:
        __tablename__ = "users"

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    user = User(id=1, name="Generated User", email="test@example.com")
    session.add(user)
    return user
