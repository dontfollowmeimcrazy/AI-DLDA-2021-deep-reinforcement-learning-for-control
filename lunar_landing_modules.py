import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander.lander == contact.fixtureA.body or self.env.lander.lander == contact.fixtureB.body:
            self.env.lander.crashed_state = True
        for i in range(2):
            if self.env.lander.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.lander.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.lander.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.lander.legs[i].ground_contact = False

class World(object):
    
    def __init__(self):
        self.world = Box2D.b2World()
        self.world.contactListener_keepref = None
        self.world.contactListener = None


    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

class LandingSite(object):
    """
    This class represents the landing site for the lander
    """
    
    def __init__(self, world, np_random=seeding.np_random(1)):
        self.world = world.world
        self.np_random = np_random
        
        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = np.random.uniform(0, H/2, size=(CHUNKS+1,))
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    ############################################################################################
    ######################## METHODS AVAILABLE IN THE LANDING SITE #############################
    ############################################################################################

    def get_position(self):
        """
        Returns the coordinates of the landing site in the 2D coordinate space that represents the moon world.
        Two coordinates are on the x axis and represent the start and end points of the landing area.
        The third coordinate is the vertical position of the site (the site is flat)
        The origin of the coordinate is in the bottom left corner of the rendered env image 
        (or you can think of a standard 2D cartesian plane with the x-axis spanning towards the right and y-axis towards above)

        Returns:
            a list of 3 floats representing the x coordinates of the start and end points of the landing site, 
            and the y coordinate of the landing site
        """
        return [self.helipad_x1, self.helipad_x2, self.helipad_y]

class LunarLander(object):
    """
    This class represents the lunar landing module
    """
    
    def __init__(self, world, np_random=seeding.np_random(1)):
        """
        Args:
            world: a World object representing the world the lander belongs to
            np_random: NumPy RandomState, the same used for the gym Env in which the lander is used
        """

        self.world = world.world
        self.np_random = np_random

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.particles = []

        self.leg_down_height = LEG_DOWN / SCALE

        self.crashed_state = False

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    ######################################################################################
    ######################## METHODS AVAILABLE IN THE LANDER #############################
    ######################################################################################

    def get_position(self):
        """
        It returns the coordinates x,y of the position of the lander 
        in the 2D coordinate space that represents the moon world
        The origin of the coordinate is in the bottom left corner of the rendered env image 
        (or you can think of a standard 2D cartesian plane with the x-axis spanning towards the right and y-axis towards above)

        Returns:
            a list of float numbers representing the x,y coordinates of the lander
        """
        #return [(self.lander.position.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2), (self.lander.position.y - (VIEWPORT_H/SCALE / 4 +LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)]
        return [self.lander.position.x, self.lander.position.y]

    def get_velocity(self):
        """
        It returns the velocity compononents (horizontal, vertical) of the lander 

        Returns:
            a list of float numbers representing the horizontal and vertical velocities of the lander
        """
        #return [self.lander.linearVelocity.x * (VIEWPORT_W/SCALE/2)/FPS, self.lander.linearVelocity.y * (VIEWPORT_W/SCALE/2)/FPS]
        return [self.lander.linearVelocity.x, self.lander.linearVelocity.y]

    def get_angle(self):
        """
        It returns the angle of the lander (in degrees) wrt to the landing surface

        Returns:
            a float number representing the angle of the lander wrt to the landing surface
        """
        return self.lander.angle

    def get_angular_velocity(self):
        """
        It returns the angular velocity of the lander wrt to the landing surface

        Returns:
            a float number representing the angular velocity of the lander wrt to the landing surface
        """
        #return 20.0 * self.lander.angularVelocity / FPS
        return self.lander.angularVelocity

    def legs_in_contact(self):
        """
        It returns the states of the terrain contact detector of each of the lunar's legs

        Returns:
            a list of bool representing the state of contact of the lander's legs,
            if the left leg is in contact with the moon terrain then at index 0 there is True, else False
            if the right leg is in contact with the moon terrain then at index 1 there is True, else False
        """
        return [self.legs[0].ground_contact, self.legs[1].ground_contact]

    def get_leg_length(self):
        """
        Return the height of the legs of the lander when in the standard position
        (this value can be summed to the y position of the lander to obtain the position of the terrain contact point)

        Returns:
            a float representing the height of the lander's legs
        """
        return self.leg_down_height

    def stable_on_terrain(self):
        """
        It returns the state of the lander after a successful landing 
        (i.e. the lander is stable on the ground and it does not move anymore)

        Returns:
            a bool representing the state in which the lander is stable on the ground
        """
        return not self.lander.awake

    def crashed(self):
        """
        It returns the state of the lander if crashed 
        (i.e. the lander contacted the ground with other parts different from the legs)
        or landed too harshly

        Returns:
            a bool representing the state in which the lander crashed
        """
        return self.crashed_state

    def fire_engine(self, action):
        """
        Method that fires the two engines of the lander. It receives in input the intensity to be applied
        to the engines (given by the lander controller, aka the agent) and uses it to compute the new 
        state (position, velocity, angle) of the lander.
        The method returns the power applied by the lander to the main and side engines.

        Args:
            action: a list of two float numbers representing the intensity to be applied to the jet engines
            of the lander. 
            The first float is the intensity for the main engines (the engine in the bottom part of the lander)
            The second float is the intensity for the left-right engines (that control the lateral movements/orientation of the lander)
            The values of the ranges have to be in the range [-1, 1] and are interepreted by the lander as follows:
                Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
                Left-right engine:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

        Returns:
            a list of floats representing the power applied by the lander to the main and side engines
        """

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle)) #sin | , cos _ , components of the angle
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Main engine
        m_power = 0.0
        if action[0] > 0.0:
            # Main engine
            m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0
            
            ox = (tip[0] * (4/SCALE + 2 * dispersion[0]) +
                  side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)

            # calculating and visualizing the fire particles
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), # impulse as N-seconds or kg-m/s
                                 impulse_pos, # psoition of the impulse
                                 True)

            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if np.abs(action[1]) > 0.5:
            # Orientation engines
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            assert s_power >= 0.5 and s_power <= 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17/SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        return [m_power, s_power]
