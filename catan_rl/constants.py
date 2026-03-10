"""Core constants and enums for the Catan RL environment."""

from enum import IntEnum


NUM_PLAYERS = 4
NUM_RESOURCES = 5
NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72
NUM_PORTS = 9
WIN_VP = 10


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


class Terrain(IntEnum):
    DESERT = 0
    WOOD = 1
    BRICK = 2
    SHEEP = 3
    WHEAT = 4
    ORE = 5


class Building(IntEnum):
    EMPTY = 0
    SETTLEMENT = 1
    CITY = 2


class Phase(IntEnum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    PRE_ROLL = 2
    DICE_ROLLED = 3
    DISCARD = 4
    MOVE_ROBBER = 5
    ROB_PLAYER = 6
    MAIN = 7
    TRADE_PROPOSED = 8
    YEAR_OF_PLENTY = 9
    MONOPOLY = 10
    ROAD_BUILDING = 11
    GAME_OVER = 12


class DevCard(IntEnum):
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VP = 4


RESOURCE_COSTS = {
    "road": [1, 1, 0, 0, 0],
    "settlement": [1, 1, 1, 1, 0],
    "city": [0, 0, 0, 2, 3],
    "dev": [0, 0, 1, 1, 1],
}


INITIAL_DEV_DECK = [14, 2, 2, 2, 5]
