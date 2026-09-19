import carla
import util
import random

if __name__ == "__main__":
    client = util.connect_to_sim()
    world = client.get_world()

    vehicles: list[carla.Actor] = []
    peds: list[carla.Actor] = []

    # Add 25 cards and 25 peds
    for i in range(0, 25):
        vehicle = util.spawn_vehicle(world)
        if not vehicle:
            print("Failed to spawn a vehicle :(")
        else:
            vehicles.append(vehicle)

    for i in range(0, 25):
        util.spawn_pedestrian(world)
        ped = util.spawn_vehicle(world)
        if not ped:
            print("Failed to spawn a pedestrian :(")
        else:
            peds.append(ped)

    # Move view to a random vehicle
    util.move_spectator_to_vehicle(world=world, vehicle=random.choice(vehicles))