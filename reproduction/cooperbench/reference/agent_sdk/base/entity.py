"""
Base Entity class for all objects in the simulation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Entity(ABC):
    """
    Abstract base class for all entities (Agents, Resources, Buildings).
    """
    def __init__(self, entity_id: str, x: int, y: int, entity_type: str):
        self.id = entity_id
        self.x = x
        self.y = y
        self.entity_type = entity_type
        self.sprite_name = "default"  # To be overridden by subclasses
        self.width = 1
        self.height = 1
        self.collision = False # Default collision state

    def to_dict(self):
        """
        Returns a dictionary representation of the entity for the API.
        """
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "type": self.entity_type,
            "sprite": self.sprite_name
        }

    @abstractmethod
    def update(self, world_state: Any):
        """
        Update the entity's state. Called once per simulation step.
        """
        pass
