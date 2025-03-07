from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

# Create fixtures and test models
Base: type = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)


class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    title = Column(String)
    content = Column(String)


# Pydantic model for testing
class UserModel(BaseModel):
    id: int
    name: str
    email: str


# Regular class for testing
class SimpleUser:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email


# Mock generator function for testing
def mock_generator(session, param1="default1", param2="default2"):
    user = User(id=1, name="Generated User", email="generated@example.com")
    session.add(user)
    return user


# Mock a function without session as first parameter
def invalid_func(param1, session):
    pass
