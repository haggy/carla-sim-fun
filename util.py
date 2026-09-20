from decimal import Decimal

import carla
import random


def connect_to_sim(host: str = "localhost", port: int = 2000) -> carla.Client:
    return carla.Client(host, 2000)

def spawn_point_or_random(world: carla.World, spawn_point: carla.Transform | None = None) -> carla.Transform:
    """
    Returns the original SP or a randomized one if not defined
    """
    return spawn_point if spawn_point is not None else random.choice(world.get_map().get_spawn_points())

def spawn_random_actor(
        world: carla.World, 
        filter_expr: str,
        spawn_point: carla.Transform | None = None
    ) -> carla.Actor | None:
    bp = random.choice(world.get_blueprint_library().filter(filter_expr))
    sp = spawn_point_or_random(world=world, spawn_point=spawn_point)
    return world.try_spawn_actor(bp, sp)

def spawn_vehicle(
        world: carla.World, 
        vehicle_bp: carla.ActorBlueprint | None = None, 
        spawn_point = None
    ) -> carla.Actor | None:
    if vehicle_bp is not None:
        return world.try_spawn_actor(blueprint=vehicle_bp, transform=spawn_point_or_random(spawn_point))
    else:
        return spawn_random_actor(world=world, filter_expr="*vehicle*", spawn_point=spawn_point)

def spawn_pedestrian(
        world: carla.World, 
        pedestrian_bp: carla.ActorBlueprint | None = None, 
        spawn_point = None
    ) -> carla.Actor | None:
    if pedestrian_bp is not None:
            return world.try_spawn_actor(blueprint=pedestrian_bp, transform=spawn_point_or_random(spawn_point))
    else:
        return spawn_random_actor(world=world, filter_expr="walker.pedestrian*", spawn_point=spawn_point)

def move_spectator_to_vehicle(
        world: carla.World, 
        vehicle: carla.Actor, 
        distance: Decimal = 6.0, 
        height: Decimal = 2.5, 
        pitch: Decimal = -15.0
    ) -> None:
    """Place the spectator camera behind and above the ego vehicle, looking at it."""
    spectator = world.get_spectator()
    vt = vehicle.get_transform()

    # Vector pointing forward from the vehicle, scaled back to sit behind it
    forward = vt.get_forward_vector()
    location = carla.Location(
        x=vt.location.x - forward.x * distance,
        y=vt.location.y - forward.y * distance,
        z=vt.location.z + height,
    )
    rotation = carla.Rotation(pitch=pitch, yaw=vt.rotation.yaw, roll=0.0)

    spectator.set_transform(carla.Transform(location, rotation))

class SensorManager:

    def __init__(self, world: carla.World):
        self._world = world
        self._sensors: list[carla.Actor] = []

    def add_camera(self, target: carla.Actor, offset: carla.Transform | None = None) -> carla.Actor:
        """
        Adds an RGB camera sensor to the specified target. If the camera offset is not specified, it defaults
        """
        # Create a transform to place the camera on top of the subject
        camera_init_trans = offset or carla.Transform(carla.Location(z=1.5))
        camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')

        # We spawn the camera and attach it to our ego vehicle
        camera = self._world.spawn_actor(camera_bp, camera_init_trans, attach_to=target)
        self._sensors.append(camera)
        return camera