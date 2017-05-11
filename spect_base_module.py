#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import math as mt
from numpy import linalg as LA
import scipy.constants as const
import warnings
import copy
import pickle


#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2
kbc = const.k/(const.h*100*const.c) # 0.69503


#####################################################################################################################
#####################################################################################################################
###############################             FUNZIONI                 ################################################
#####################################################################################################################
#####################################################################################################################
class Coords(object):
    """
    Coordinates. Some useful trick.
    """
    def __init__(self, coords, s_ref = 'Cartesian', R = Rtit):
        """
        Coordinate order is as follows:
        #Cartesian# : (x,y,z)
        #Spherical# : (lat,lon,R)
        #Pure_spherical# : (R, theta, phi)
        """
        self.coords = np.array(coords, dtype = float)
        self.s_ref = s_ref
        if s_ref == 'Cartesian':
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
        elif s_ref == 'Spherical':
            self.lat = coords[0]
            self.lon = coords[1]
            self.height = coords[2]
            self.R = R
        elif s_ref == 'Pure_spherical':
            self.R = coords[0]
            self.theta = coords[1]
            self.phi = coords[2]
        else:
            raise ValueError('Reference system of type {}'.format(s_ref))

        return

    def Spherical(self):
        if self.s_ref == 'Cartesian':
            return carttosph(self.coords)
        elif self.s_ref == 'Spherical':
            return self.coords

    def Cartesian(self):
        if self.s_ref == 'Cartesian':
            return self.coords
        elif self.s_ref == 'Spherical':
            return sphtocart(self.coords, R=self.R)


class LineOfSight(object):
    """
    Class to represent the geometry of an observation.
    """

    def __init__(self, spacecraft_coords, second_point):
        self.starting_point = copy.deepcopy(spacecraft_coords)
        self.second_point = copy.deepcopy(second_point)
        print(self.starting_point.Cartesian(),self.second_point.Cartesian())
        print(self.starting_point.Cartesian(),self.second_point.Cartesian())
        print(self.starting_point.Cartesian(),self.second_point.Cartesian())
        print(self.starting_point.Cartesian(),self.second_point.Cartesian())

        return

    def calc_LOS_vector(self):#, planet, ellipsoidal = False):
        """
        Adds to the LOS object a parameter vector that points from the Spacecraft to the observed point on the planet.
        """

        c1 = copy.deepcopy(self.starting_point.Cartesian())
        c2 = copy.deepcopy(self.second_point.Cartesian())

        _LOS = (c2-c1)/LA.norm(c2-c1) # Unit LOS vector
        print(c2-c1,LA.norm(c2-c1))
        print(self.starting_point.Cartesian(),self.second_point.Cartesian())
        self._LOS = _LOS

        return _LOS


    def details(self):
        print('Observer point: {}, {}'.format(self.starting_point.Cartesian(), self.starting_point.Spherical()))
        print('Second point: {}, {}'.format(self.second_point.Cartesian(), self.second_point.Spherical()))
        try:
            print('LOS vector: {}'.format(self._LOS))
        except:
            print('LOS vector: {}'.format(self.calc_LOS_vector()))

        return

    def calc_along_LOS(self, atmosphere, profname, curgod = False, set_attr = False, set_attr_name = None):
        """
        Returns a vector with prof values on the LOS.
        ###################### WIPPPPPP
        CURGOD TO BE INTRODUCED
        method available: simple, curgod
        """

        try:
            points = self.intersections
        except:
            points = self.calc_atm_intersections(planet)

        self.atm_quantities = dict([])

        quant = []

        for point1,point2 in zip(points[:-1],points[1:]):
            point = Coords((point1.Cartesian()+point2.Cartesian())/2)
            quant.append(atmosphere.calc(point.Spherical()[2],profname))
            if curgod:
                print('Not yet available! Doing simple interpolation..')

        if set_attr:
            if set_attr_name is None:
                set_attr_name = profname
            self.atm_quantities[set_attr_name] = np.array(quant)

        return np.array(quant)


    def calc_SZA_along_los(self,planet,sub_solar_point):
        try:
            points = self.intersections
        except:
            points = self.calc_atm_intersections(planet)

        szas = []
        for point in points:
            szas.append(angle_between(point.Cartesian(), sub_solar_point.Cartesian()))

        self.szas = np.array(szas)

        return np.array(szas)


    def calc_atm_intersections(self, planet, delta_x = 5.0, start_from_TOA = True, extinction_coeff = None, refraction = False, adaptive_step = False):
        """
        Calculates the coordinates of the points that the LOS crosses in the atmosphere. If extinction_coeff is set, the coordinate along the LOS is tau, the optical depth, instead it is just the distance in km. If start_from_TOA is True, the LOS path starts at the first intersection with the atmosphere.

        NO REFRACTION INCLUDED!! work in progress.....
        """

        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            print(cazzillo)

        TOA_R = planet.atm_extension + planet.radius
        R = planet.radius

        if start_from_TOA:
            print('Qui')
            print(self.starting_point.Cartesian())
            point_0, i_type = self.intersect_shell(planet, self.starting_point.Cartesian(), TOA_R)
            if i_type != 'Ingress':
                raise ValueError('No intersection with shell!')
            print('Quo')
        else:
            point_0 = copy.deepcopy(self.starting_point.Cartesian())


        min_R = self.get_closest_distance()
        if min_R <= R:
            self.nadir_flag = True
            self.limb_flag = False
            self.deepsky_flag = False
            self.tangent_altitude = self.min_distance-R
        elif min_R > R and min_R <= TOA_R:
            self.nadir_flag = False
            self.limb_flag = True
            self.tangent_altitude = self.min_distance-R
            self.deepsky_flag = False
        else:
            self.nadir_flag = False
            self.limb_flag = False
            self.deepsky_flag = True
            self.tangent_altitude = self.min_distance-R

        los_points = []
        los_points.append(Coords(point_0, R = planet.radius))

        inside = True
        point = point_0
        while inside:
            if adaptive_step:
                #Something that calculates the best step at point
                pass
            point = self.move_along_LOS(planet, point, delta_x, refraction = refraction)
            inside = (LA.norm(point) < TOA_R)
            print(inside, point)
            surface_hit = (LA.norm(point) < R)
            if surface_hit:
                point = self.move_along_LOS(planet, point, delta_x, refraction = refraction, backward = True)
                point, i_type = self.intersect_shell(planet, point, R)
                if i_type != 'Ingress':
                    raise ValueError('No intersection with shell!')
                los_points.append(Coords(point, R = planet.radius))
                print('Hit the surface.. Stopping ray path.')
                break
            elif inside and not surface_hit:
                print('Still inside')
                los_points.append(Coords(point, R = planet.radius))

        los_points = los_points[::-1]

        self.intersections = los_points

        return los_points


    def intersect_shell(self, planet, point, shell_radius, thres = 0.1, refraction = False):
        """
        Starting from point, finds the closer intersection with shell at radius shell_radius.
        DA AGGIUNGERE! : controllo di esistenza dell'intersezione
        # Ho 3 casi.
        # 1 - C'è almeno una intersezione nella forward direction, do come output la più vicina, che sia in ingress o egress.
        # 2 - Ci sono intersezioni solo in backward direction. Metto una flag backward e quando è falsa ritorno not found. Forse questa non è utile. Però vorrei che mi desse un output diverso da 3.
        # 3 - La LOS non interseca mai la shell. Not found.
        """

        print('Starting from: {}'.format(point))
        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            print(cazzillo)

        # Look for the closest point to origin.
        min_R = self.get_closest_distance(refraction = refraction)
        deltino = LA.norm(point)-shell_radius

        if min_R > shell_radius:
            print('LOS does not intersect shell. Tangent altitude is {} at point {}.'.format(min_R,self.tangent_point.Spherical()))
            i_type='No intersection'
            return np.array(3*[np.nan]),i_type
        elif min_R <= shell_radius and deltino < 0:
            print('Point {} is already inside shell at radius {}. Are you looking for an egress intersection? Yet not available----zorry!'.format(Coords(point).Spherical(),shell_radius))
            i_type='Egress'
            return np.array(3*[np.nan]),i_type
        else:
            while deltino > thres:
                deltino = LA.norm(point)-shell_radius
                deltino = deltino/LA.linalg.dot(-normalize(point), _LOS)
                point = self.move_along_LOS(planet, point, deltino, refraction = refraction)
            i_type = 'Ingress'
            return point, i_type


    def move_along_LOS(self, planet, point, step, refraction = False, backward = False):
        """
        Simply makes a step in the los. PLANNED INCLUSION OF REFRACTION.
        """

        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            print(cazzillo)

        if not backward:
            point += _LOS * step
        else:
            point -= _LOS * step

        return point


    def get_tangent_point(self, point_0 = np.array([0,0,0]),refraction = False):
        """
        Returns the point of LOS which is closest to the origin.
        """
        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            print(cazzillo)

        if not refraction:
            t_point = tangent_point(_LOS, self.starting_point.Cartesian(), point = point_0)
            self.tangent_point = Coords(t_point)
        else:
            pass
            raise ValueError('Not yet available with refraction')

        return Coords(t_point)


    def get_closest_distance(self, point_0 = np.array([0,0,0]),refraction = False):
        """
        Returns the point of LOS which is closest to the origin.
        """

        t_point = self.get_tangent_point(point_0,refraction)
        dist = LA.norm(t_point.Cartesian()-point_0)
        self.min_distance = dist

        return dist


def normalize(vector):
    norm_vector = vector/LA.norm(vector)
    return norm_vector


def angle_between(vector1,vector2,deg_output=True):
    """
    Returns the angle between vectors. Default output in deg.
    """
    v1 = normalize(vector1)
    v2 = normalize(vector2)

    angle = mt.acos(LA.linalg.dot(v1,v2))

    if deg_output:
        return deg(angle)
    else:
        return angle


def orthogonal_plane(vector, point):
    line = normalize(vector)

    if line[2] != 0:
        print('Returning plane as function of (x,y) couples')
        def plane(x,y):
            z = point[2] - ( line[0]*(x-point[0]) + line[1]*(y-point[1]) ) / line[2]
            return np.array([x,y,z])
    else:
        print('Returning plane as function of (x,z) couples')
        def plane(x,z):
            y = point[1] - line[0] * (x-point[0]) / line[1]
            return np.array([x,y,z])

    return plane


def tangent_point(vector, r_ini, point = np.array([0,0,0])):
    """
    For a given LOS, returns the point of minimum distance from the origin or the selected point.
    """

    t_point = t_crit_line_point(vector, r_ini, point) * normalize(vector) + r_ini

    return t_point


# def t_crit_line_point(vector, r_ini, point = np.array([0,0,0])):
def distance_line_origin(vector, r_ini, point = np.array([0,0,0])):
    """
    Returns the distance between the point and the span of the vector. Origin is the standard point.
    """

    t_point = tangent_point(vector, r_ini, point)
    dist = LA.norm(t_point-point)

    return dist


def t_crit_line_point(vector, r_ini, point = np.array([0,0,0])):
    """
    Ausiliary to distance_line_origin.
    """
    line = normalize(vector)
    ortholine = orthogonal_plane(line, point)

    tc = - LA.linalg.dot(line,(r_ini-point))

    return tc



class Planet(object):
    """
    Class to represent a planet.
    """

    def __init__(self, name, color = None, planet_radius = None, planet_radii = None, planet_mass = None, planet_type = None, atm_extension = None, planet_star_distance = None, rotation_axis_inclination = None, rotation_axis = None, rotation_period = None, revolution_period = None, obs_time = None, is_satellite = False, orbiting_planet = None, satellite_planet_distance = None, satellite_orbital_period = None):
        self.name = name
        self.color = color # representative color (just for drawing purposes)
        self.radius = planet_radius # MEAN radius
        self.radii = planet_radii # LIST with EQUATORIAL and POLAR radius
        self.mass = planet_mass
        self.type = planet_type # Terrestrial, Gas Giant, ...
        self.atm_extension = atm_extension # mainly for drawing purposes
        self.planet_star_distance = planet_star_distance
        self.rotation_axis = rotation_axis # THis is a vector. the reference frame is: z -> perpendicular to the orbit plane; x -> points from star to planet;
        self.rotation_axis_inclination = rotation_axis_inclination
        self.rotation_period = rotation_period
        self.revolution_period = revolution_period
        self.obs_time = obs_time # If I create a planet at some specific season

        self.is_satellite = is_satellite
        self.is_satellite_of = orbiting_planet
        self.satellite_planet_distance = satellite_planet_distance
        self.satellite_orbital_period = satellite_orbital_period

        return


    def add_atmosphere(self, atmosphere):
        """
        Creates a link to an AtmProfile object containing temp, pres, ...
        """

        if type(atmosphere) is not AtmProfile:
            raise ValueError('atmosphere is not an AtmProfile object. Create it via Prof_ok = AtmProfile(atmosphere, grid).')

        self.atmosphere = atmosphere
        self.gases = dict([])

        return

    def add_gas(self, gas):
        """
        Creates a link to an AtmProfile object containing vibtemps ...
        """

        self.gases[gas.name] = gas
        print('Gas {} added to planet.gases.'.format(gas.name))

        return


class Titan(Planet):
    def __init__(self):
        Planet.__init__(self,name='Titan', color = 'orange', planet_type = 'Terrestrial', planet_mass = 0.0225, planet_radius = 2575., planet_radii = [2575., 2575.], atm_extension = 1000., is_satellite = True, orbiting_planet = 'Saturn', satellite_orbital_period = 15.945, satellite_planet_distance = 1.222e6, rotation_period = 15.945, rotation_axis_inclination = 26.73, revolution_period = 29.4571, planet_star_distance = 9.55)

        return

    def add_default_atm(self):
        cart2 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Test_wave2/'

        ch4_nom, ch4_wave2 = pickle.load(open(cart2+'ch4_Molec_testMaya.pic','r'))

        Planet.add_atmosphere(self,ch4_nom.atmosphere)
        Planet.add_gas(self,ch4_nom)

        return





class Pixel(object):
    """
    Each instance is a pixel, with all useful things of a pixel. Simplified version with few attributes.
    """

    def __init__(self, keys = None, things = None):
        if keys is None:
            return
        for key, thing in zip(keys,things):
            setattr(self,key,thing)
        # self.cube = cube
        # self.year = year
        # self.dist = dist
        # self.lat = lat
        # self.alt = alt
        # self.sza = sza
        # self.phang = phang
        # self.wl = np.array(wl)
        # self.spe = np.array(spe)
        # self.bbl = np.array(bbl)
        return

    def plot(self, range=None, mask=True, show=True, nomefile=None):
        """
        Plots spectrum in the required range. If mask=True does not show masked values.
        :param range: Wl range in which to plot
        :return:
        """
        if range is not None:
            ok = (self.wl >= range[0]) & (self.wl <= range[1]) & (self.mask == 1)
        else:
            ok = (self.mask == 1)

        fig = pl.figure(figsize=(8, 6), dpi=150)
        pl.plot(self.wl[ok],self.spe[ok])
        pl.grid()
        pl.xlabel('Wavelength (nm)')
        pl.ylabel('Radiance (W/m^2/nm/sr)')
        if show: pl.show()
        if nomefile is not None:
            fig.savefig(nomefile, format='eps', dpi=150)
            pl.close()

        return

    def integr(self, range=None, fondo = 0.0):
        """
        Integrates the spectrum in the selected range. If no range the full spectrum is integrated.
        """

        if range is None:
            range = [np.min(self.wl),np.max(self.wl)]

        cond = (self.wl >= range[0]) & (self.wl <= range[1]) & (self.mask == 1)
        intt = np.trapz(self.spe[cond]-fondo,x=self.wl[cond])

        return intt



class PixelSet(np.ndarray):
    """
    A set of pixels. Takes as input an existing array of pixels and adds as attributes the vectorized attributes of Pixel.
    """

    def __new__(cls, input_array, descr=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.descr = descr
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.descr = getattr(obj, 'descr', None)
        for name in obj[0].__dict__.keys():
            try:
                setattr(self,name,np.array([getattr(pix,name) for pix in self]))
            except:
                setattr(self,name+'_ok',np.array([getattr(pix,name) for pix in self]))
        return

    def ciao(self):
        print('ciao')
        return

    def integr_tot(self, key=None, **kwargs):
        intg = np.array([pix.integr(**kwargs) for pix in self])
        if key is not None:
            setattr(self,key,intg)
            for pix,intg1 in zip(self,intg):
                setattr(pix,key,intg1)
        return intg


class Molec(object):
    """
    Class to represent molecules, composed by isoMolecules.
    """

    def __init__(self, mol, name, MM=None):
        self.mol = mol   # mol number
        self.name = name # string representing the molecular name
        self.MM = MM   # average molecular mass
        self.abundance = None
        self.n_iso = 0
        self.all_iso = []
        return

    def add_clim(self, profile, type = 'VMR'):
        """
        Defines a climatology for the molecule. profile is the molecular abundance, has to be an AtmProfile object. type can be either 'VMR' or 'num_density (cm-3)'
        """
        if type(profile) is not AtmProfile:
            raise ValueError('Profile is not an AtmProfile object. Create it via Prof_ok = AtmProfile(profile, grid).')
        setattr(self, 'abundance', profile)
        print('Added climatology for molecule {}'.format(self.name))

        return

    def add_iso(self, num, MM=None, ratio = None):
        """
        Adds an isotope in form of an IsoMolec object.
        """
        string = 'iso_{:1d}'.format(num)
        iso = IsoMolec(self.mol,num,MM=MM,ratio=ratio)
        setattr(self, string, iso)
        print('Added isotopologue {} for molecule {}'.format(string,self.name))
        self.n_iso += 1
        self.all_iso.append(string)
        return

    def link_to_atmos(self, atmosphere, copy = False):
        """
        Creates a link to an AtmProfile object containing temp, pres, ...
        """
        if not copy:
            self.atmosphere = atmosphere
        else:
            raise Exception('DEVO ANCORA SCRIVERLO!')

        return


class IsoMolec(object):
    """
    Class to represent isotopes, with iso-ratios, vibrational levels, vib. temperatures, (for now).
    """

    def __init__(self, mol, iso, MM=None, ratio=None):
        self.mol = mol   # mol number
        self.iso = iso   # iso number
        self.MM = MM   # mol mass
        self.ratio = ratio   # isotopic ratio
        self.n_lev = 0 # number of levels specified
        self.levels = []
        return

    def add_levels(self, lev_strings, energies, vibtemps = None, degeneracies = None, simmetries = None):
        """
        Adds vibrational levels to selected isotopologue.
        """

        if degeneracies is None:
            degeneracies = len(lev_strings)*[-1]
        if simmetries is None:
            simmetries = len(lev_strings)*[[]]
        if vibtemps is None:
            vibtemps = len(lev_strings)*[None]

        for levstr,ene,deg,sim,i,vib in zip(lev_strings,energies,degeneracies,simmetries,range(len(lev_strings)),vibtemps):
            print('Level <{}>, energy {} cm-1'.format(levstr,ene))
            string = 'lev_{:02d}'.format(i)
            self.levels.append(string)
            lev = Level(levstr, ene, degeneracy = deg, simmetry = sim)
            if vib is not None:
                lev.add_vibtemp(vib)
            setattr(self,string,lev)
            self.n_lev += 1

        return

    def add_level(self, lev_string, energy, degeneracy = -1, simmetry = [''], vibtemp = None):
        """
        Adds a single vib. level.
        """
        print('Level <{}>, energy {} cm-1'.format(lev_string,energy))
        string = 'lev_{:02d}'.format(self.n_lev)
        lev = Level(lev_string, energy, degeneracy = degeneracy, simmetry = simmetry)
        if vibtemp is not None:
            lev.add_vibtemp(vibtemp)
        setattr(self,string,lev)
        self.levels.append(string)
        self.n_lev += 1

    def find_level_from_quanta(self, quanta, simmetry = None):
        """
        Given the set of quantum numbers (list) returns the level object. Optionally also simmetry can be set.
        """

        found = False
        for lev in self.levels:
            level = getattr(self, lev)
            qlev = level.get_quanta(self.mol)
            if simmetry is None:
                if qlev[0] == quanta:
                    found = True
            else:
                if qlev[0] == quanta and qlev[1] == simmetry:
                    found = True

            if found: break

        if found:
            return level
        else:
            print('Level {} with simmetry {} not found.'.format(quanta,simmetry))
            return None


class Level(object):
    """
    Class to represent one single vibrational level of a molecule. Contains HITRAN strings, energy, degeneration.
    Planned: decomposition of HITRAN strings in vib quanta.
    """

    def __init__(self, levstring, energy, degeneracy = -1, simmetry = []):
        self.lev_string = levstring
        self.energy = energy
        self.degeneracy = degeneracy
        self.simmetry = simmetry
        self.vibtemp = None
        return

    def add_vibtemp(self,profile):
        """
        Add the vibrational temperature of the level.
        """
        if type(profile) is not AtmProfile:
            raise ValueError('Profile is not an AtmProfile object. Create it via Prof_ok = AtmProfile(profile, grid).')
        setattr(self, 'vibtemp', profile)
        print('Added vibrational temperature to level <{}>'.format(self.lev_string))

        return

    def get_quanta(self, mol):
        """
        Reads from the lev_string the quantum numbers and returns a list plus a simmetry. Need to specify the molecule for now. Output format: ([n1, n2, n3, ..], simmetry)
        """
        if mol == 6:
            quanta = extract_quanta_ch4(self.lev_string)
        else:
            raise ValueError('No routine available for molecule {}'.format(mol))

        return quanta


class CosaScema(np.ndarray):
    """
    Sottoclasse scema di ndarray.
    """
    def __new__(cls, profile, ciao = 'ciao'):
        print(1, 'dentro new')
        obj = np.asarray(profile).view(cls)
        print(4,type(obj),obj.ciao)
        try:
            print(4.1,type(self),self.ciao)
        except Exception as stup:
            print(4.1,stup)
        obj.ciao = ciao
        print(5,type(obj),obj.ciao)
        return obj

    def __array_finalize__(self,obj):
        print(2, 'dentro array_finalize')
        if obj is None: return
        self.ciao = getattr(obj,'ciao',None)
        print(3, type(self), type(obj))
        print(3.1, self.ciao, getattr(obj,'ciao',None))
        return


class AtmProfile(object):
    """
    Class to represent atmospheric profiles. Contains a geolocated grid and a routine to get the interpolated value of the profile in the grid.
    Contains a set of interp strings to determine the type of interpolation.
    Input:
        grid -----------> coordinates grid : np.mgrid with variable dimensions. es: (lat,lon,alt), (lat,alt,sza). The dimension of the mgrid object is the grid dimension + 1.
        gridname -------> names for the dimensions of grid
        profile --------> profile : dim of grid - 1
        profname ----------> name of the profile (es. 'temp', 'pres', ..)
        interp ---------> list of strings : how to interpolate in given dimension? Accepted values: 'lin' - linear interp, 'exp' - exponential interp o 'box' - nearest neighbour o ... others ...
        hierarchy ------> smaller value indicates in which dimension to interpolate first? if 2 dimensions have the same value

    Attributes: names (list of profiles names -> list of strings), [nam for nam in names] (all profiles -> ndarrays), grid (the grid -> np.mgrid), gridname (names of the grid coords -> list of strings), interp (list of interp lists, one per profile), hierarchy (list od hierarchies, one per profile), descr
    """

    def __init__(self, profile, grid, profname = 'prof', gridname = None, interp=None, hierarchy = None, descr=None):

        profile = np.array(profile)
        grid = np.array(grid)
        self.descr = descr
        setattr(self, profname, profile.copy())
        self.ndim = profile.ndim

        profnames = []
        profnames.append(profname)
        self.names = profnames

        if grid.ndim == 1:
            grid = np.array([grid])
            print('Transformed grid to a 1-dim grid array',np.shape(grid))

        self.grid = grid.copy()
        self.interp = []
        self.hierarchy = []

        if self.ndim != grid.ndim-1: raise ValueError('profile and grid have different dimensions!')

        if gridname is not None:
            self.gridname = np.array(gridname)
            for name,coord in zip(gridname,grid):
                setattr(self,name,np.sort(np.unique(coord)))
        else:
            self.gridname = np.array(['']*self.ndim)
            print('No gridname set. Enter names for the coordinate grid')

        if interp is not None:
            for ino in interp:
                nam = ['lin','exp','box']
                try:
                    nam.index(ino)
                except ValueError:
                    raise ValueError('{} is not a possible interpolation scheme. Possible schemes: {}'.format(ino,nam))
            self.interp.append(np.array(interp))
        else:
            interp = np.array(['lin']*self.ndim)
            if profname == 'pres':
                indx = [('Alt' in pi or 'alt' in pi) for pi in self.gridname].index(True)
                interp[indx] = 'exp'
            self.interp.append(np.array(interp))
            print('No interp found, interp set to default: {}'.format(self.interp[0]))

        if hierarchy is not None:
            self.hierarchy.append(np.array(hierarchy))
        else:
            self.hierarchy.append(np.arange(self.ndim))
            print('No hierarchy found, hierarchy set to default: {}'.format(self.hierarchy[0]))
        for val in np.unique(self.hierarchy[0]):
            oi = (self.hierarchy[0] == val)
            if len(np.unique(self.interp[0][oi])) > 1:
                raise ValueError('Can not interpolate 2D between int and exp coordinates!')
            elif np.unique(self.interp[0][oi])[0] == 'exp' and len(self.interp[0][oi]) > 1:
                raise ValueError('Can not interpolate 2D between exp coordinates!')

        return

    def add_profile(self, profile, profname, gridname = None, interp=None, hierarchy = None):
        """
        Adds a new profile to the atmosphere.
        """
        profile = np.array(profile)

        if self.ndim != profile.ndim: raise ValueError('New profile {} is not consistent with {}!'.format(profname, self.names[0]))
        setattr(self, profname, profile.copy())

        self.names.append(profname)

        if interp is not None:
            for ino in interp:
                nam = ['lin','exp','box']
                try:
                    nam.index(ino)
                except ValueError:
                    raise ValueError('{} is not a possible interpolation scheme. Possible schemes: {}'.format(ino,nam))
            interp = np.array(interp)
            self.interp.append(interp)
        else:
            interp = np.array(['lin']*self.ndim)
            if profname == 'pres':
                indx = [('Alt' in pi or 'alt' in pi) for pi in self.gridname].index(True)
                interp[indx] = 'exp'
            self.interp.append(interp)
            print('No interp found, interp set to default: {}'.format(interp))

        if hierarchy is not None:
            hierarchy = np.array(hierarchy)
            self.hierarchy.append(hierarchy)
        else:
            hierarchy = np.arange(self.ndim)
            self.hierarchy.append(hierarchy)
            print('No hierarchy found, hierarchy set to default: {}'.format(hierarchy))
        for val in np.unique(hierarchy):
            oi = (hierarchy == val)
            if len(np.unique(interp[oi])) > 1:
                raise ValueError('Can not interpolate 2D between int and exp coordinates!')
            elif np.unique(interp[oi])[0] == 'exp' and len(interp[oi]) > 1:
                raise ValueError('Can not interpolate 2D between exp coordinates!')

        return

    # def __getitem__(self, *args, **4kwargs):
    #     for name in self._all_:
    #         profilo = getattr(self, name)
    #         sliced_prof = profilo.__getitem__(*args,**kwargs)
    #         setattr(obj, name, sliced_prof)
    #     np.ndarray.__getitem__(self.array_view, *args, **kwargs)
    #     return
    #
    # def __getslice__(self, *args, **kwargs):
    #     for name in self._all_:
    #         profilo = getattr(self, name)
    #         sliced_prof = profilo.__getslice__(*args,**kwargs)
    #         setattr(obj, name, sliced_prof)
    #     obj.grid = self.grid
    #     return obj


    def calc(self, point, profname = None):
        """
        Interpolates the profile at the given point.
        :param point: np.array point to be considered. len(point) = self.ndim
        :param profname: name of the profile to calculate (es: 'temp', 'pres')
        :return: If only one profile is stored in AtmProfile, the output is a number. If there are more profiles stored in AtmProfile, the output of calc is a dict containing all interpolated values. If profname is set (ex. 'temp') returns the value of the profile 'temp' at the point.
        """
        resu = []
        for nam, i in zip(self.names, range(len(self.names))):
            prof = getattr(self, nam)
            value = interp(prof, self.grid, point, itype=self.interp[i], hierarchy=self.hierarchy[i])
            resu.append(value)

        if len(self.names) == 1:
            return resu[0]
        elif profname is not None:
            dicto = dict(zip(self.names,resu))
            return dicto[profname]
        else:
            return dict(zip(self.names,resu))


    def interp_copy(self, nomeprof, new_grid):
        """
        Interpolates the original profile nomeprof in the new_grid coords. Returns the interpolated profile.
        """

        if new_grid.ndim == 1:
            new_grid = np.array([new_grid])

        new_prof = []
        for point in zip(*[luii.flat for luii in new_grid]):
            new_prof.append(self.calc(point, nomeprof))

        return np.array(new_prof)


    def collapse(self, coords):
        """
        Returns the profile corresponding to the selected coordinates. If all coordinates are specified returns the scalar value from calc.
        :param coords:
        :return:
        """
        pass

    def smooth_prof(self, points):
        """
        Returns a smoothed profile, with spline interpolation. To be used mainly for showing purposes.
        """
        pass

    def calc_along_LOS(self, LOS, nomeprof = 'prof', step = None, tau_step = 1e-5, max_line_strength = None, delta_var_prof = 0.01):
        """
        Returns a vector with prof values on the LOS and a vector of positions of the selected points.
        """
        pass



#### FINE AtmProfile class

class AtmProfile_ndarr(np.ndarray):
    """
    Class to represent atmospheric profiles. Contains a geolocated grid and a routine to get the interpolated value of the profile in the grid.
    Contains a set of interp strings to determine the type of interpolation.
    Input:
        grid -----------> coordinates grid : np.mgrid with variable dimensions. es: (lat,lon,alt), (lat,alt,sza) ..
        gridname -------> names for the dimensions of grid
        profile --------> profile : same dimensions as grid
        interp ---------> list of strings : how to interpolate in given dimension? Accepted values: 'lin' - linear interp, 'exp' - exponential interp o 'box' - nearest neighbour o ... others ...
        hierarchy ------> smaller value indicates in which dimension to interpolate first? if 2 dimensions have the same value
    """

    def __new__(cls, profile, grid, profname = 'prof', gridname = None, interp=None, hierarchy = None, descr=None):

        obj = np.asarray(profile).view(cls)

        obj.descr = descr
        setattr(obj, profname, profile.copy())

        _all_ = []
        _all_.append(profname)
        obj._all_ = _all_

        if grid.ndim == 1:
            grid = np.array([grid])
            print('Transformed grid to a 1-dim grid array',np.shape(grid))

        obj.grid = grid.copy()

        if obj.ndim != grid.ndim-1: raise ValueError('profile and grid have different dimensions!')

        if interp is not None:
            for ino in interp:
                nam = ['lin','exp','box']
                try:
                    nam.index(ino)
                except ValueError:
                    raise ValueError('{} is not a possible interpolation scheme. Possible schemes: {}'.format(ino,nam))
            obj.interp = np.array(interp)
        else:
            obj.interp = np.array(['lin']*obj.ndim)
            print('No interp found, interp set to default: {}'.format(obj.interp))

        if gridname is not None:
            obj.gridname = np.array(gridname)
            for name,coord in zip(gridname,grid):
                setattr(obj,name,np.sort(np.unique(coord)))
        else:
            obj.gridname = np.array(['']*obj.ndim)
            print('No gridname set. Enter names for the coordinate grid')

        if hierarchy is not None:
            obj.hierarchy = np.array(hierarchy)
        else:
            obj.hierarchy = np.arange(obj.ndim)
            print('No hierarchy found, hierarchy set to default: {}'.format(obj.hierarchy))
        for val in np.unique(obj.hierarchy):
            oi = (obj.hierarchy == val)
            if len(np.unique(obj.interp[oi])) > 1:
                raise ValueError('Can not interpolate 2D between int and exp coordinates!')
            elif np.unique(obj.interp[oi])[0] == 'exp':
                raise ValueError('Can not interpolate 2D between exp coordinates!')

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.descr = getattr(obj, 'descr', None)
        self.grid = getattr(obj, 'grid', None)
        self.prof = getattr(obj, 'prof', None)
        self._all_ = getattr(obj, '_all_', None)
        self.interp = getattr(obj, 'interp', None)
        self.hierarchy = getattr(obj, 'hierarchy', None)
        return

    # def __getitem__(self, *args, **kwargs):
    #     for name in self._all_:
    #         profilo = getattr(self, name)
    #         sliced_prof = profilo.__getitem__(*args,**kwargs)
    #         setattr(obj, name, sliced_prof)
    #     np.ndarray.__getitem__(self.array_view, *args, **kwargs)
    #     return
    #
    # def __getslice__(self, *args, **kwargs):
    #     for name in self._all_:
    #         profilo = getattr(self, name)
    #         sliced_prof = profilo.__getslice__(*args,**kwargs)
    #         setattr(obj, name, sliced_prof)
    #     obj.grid = self.grid
    #     return obj

    def calc(self, point):
        """
        Interpolates the profile at the given point.
        :param point: np.array point to be considered. len(point) = self.ndim
        :return:
        """

        value = interp(self, self.grid, point, itype=self.interp, hierarchy=self.hierarchy)

        return value

    def collapse(self, coords):
        """
        Returns the profile corresponding to the selected coordinates. If all coordinates are specified returns the scalar value from calc.
        :param coords:
        :return:
        """
        pass

    def smooth_prof(self, points):
        """
        Returns a smoothed profile, with spline interpolation. To be used mainly for showing purposes.
        """
        pass


#### FINE AtmProfile class

def interp(prof, grid, point, itype=None, hierarchy=None):
    """
    Returns the a matrix of dims ndim x 2, with indexes of the two closest values among the grid coordinates to the point coordinate.
    :param point: 1d array with ndim elements, the point to be considered
    :param grid: np.mgrid() array (ndim+1)
    :param prof: the profile to be interpolated (ndim array)
    :param itype: type of interpolation in each dimension, allowed values: 'lin', 'exp', 'box', '1cos'
    :param hierarchy: order for interpolation
    :return:
    """

    try:
        ndim = len(point)
    except: # point è uno scalare
        ndim = 1
        point = np.array([point])

    if ndim != grid.ndim-1:
        raise ValueError('Point should have {:1d} dimensions, instead has {:1d}'.format(grid.ndim,ndim))

    if itype is None:
        itype = np.array(ndim*['lin'])
    if hierarchy is None:
        hierarchy = np.arange(ndim)

    # Calcolo i punti di griglia adiacenti, con i relativi pesi
    indxs = []
    weights = []
    #print(np.shape(grid))
    for p, arr, ity in zip(point,grid,itype):
        #print(p, arr, ity)
        indx, wei = find_between(np.unique(arr),p,ity)
        indxs.append(indx)
        weights.append(wei)

    #print(indxs,weights)
    #print(np.shape(prof))
    # mi restringo l'array ad uno con solo i valori che mi interessano
    profi = prof
    for i in range(ndim):
        profi = np.take(profi,indxs[i],axis=i)
        #gridi = np.take(gridi,indxs[i],axis=i)
    #print(np.shape(profi))
    #print(profi)

    # Calcolo i valori interpolati, seguendo hierarchy in ordine crescente
    hiesort = np.argsort(hierarchy)
    profint = profi
    for ax in hiesort:
        vals = [np.take(profint,ind,axis=ax) for ind in [0,1]]
        profint = int1d(vals,weights[ax],itype[ax])

    return profint

    # ok, tu vuoi aggiornare grid_weights con le nuove weights? o già dargli un valore interpolato? la seconda? No, aspaspasp.
    # Allora per le dimensioni che tratto insieme va bene avere un unico schema di grid_weights. MA SE i grid_weights restassero separati nelle diverse dimensioni,
    # cioè se avesse la dimensione di grid? così li calcolo tutti ad una volta e mi salto un sacco di cazzzzzzi, oh yeah. Poi fuori da questa funzione capisco come fare!

    #sono arrivato quiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

    # adesso voglio che value sia una matr con le dim di profile che ha tutto 0 tranne nei punti coinvolti nella media dove ha il peso relativo
    # ora per adesso posso lasciare degli 1 nei punti coinvolti e mano a mano che interpolo aggiorno i pesi
    # cioè ad ogni livello di hierarchy moltiplico i punti per il peso che hanno nel nuovo livello
    # mm ancora un po' confuso.
    # cioè da qui devono uscire gli indici dei punti coinvolti


    ##### MI sono incartato nel punto in cui cercavo di capire che cazzo voglio come output di sta funzione di merda

    # meglio fare un where che funziona benissimo
    # tanto gli step sono fissi no diocane eh vabbè però pure te....
    # basta fare un where con la distanza dal punto per ogni coord minore di step/2
    # vabbè insomma mi trovo sti due punti, e ok. poi i valori delle griglie sopra e sotto li uso nella ricerca e estraggo una sottomatrice
    #
    # In[242]: cond = ((mgr[0,] == 80) | (mgr[0,] == 90)) & ((mgr[1,] == 110) | (mgr[1,] == 120))
    # In[243]: vec[cond]
    # Out[236]: array([ 320.,  320.,  360.,  360.])
    # In[244]: mgr[0,cond]
    # Out[237]: array([ 80.,  80.,  90.,  90.])
    # In[245]: mgr[1,cond]
    # Out[238]: array([ 110.,  120.,  110.,  120.])


def vibtemp_to_ratio(energy,vibtemp,temp):
    """
    Converts vibrational temperature profile to non-lte ratio. Simple array profiles as input.
    """

    prof = np.exp(-energy/kbc*(1/vibtemp-1/temp))

    return prof


def ratio_to_vibtemp(energy,ratio,temp):
    """
    Converts non-lte ratio to vibrational temperature profile. Simple array profiles as input.
    """

    prof = energy/kbc*(energy/(kbc*temp)-np.log(ratio))**(-1)

    return prof


def int1d(vals,weights,itype):
    """
    Given the two values with relative weights, gives the interpolated value according to itype.
    :param vals: list[2] the two objects: can be scalars or ndarrays
    :param weights: list[2] the two weights
    :param itype: 'lin','exp','box','1cos'
    :return: intval, the interpolated value
    """

    if itype == 'lin' or itype == '1cos' or itype == 'box': # linear interpolation
        intval = weights[0]*vals[0]+weights[1]*vals[1]
    elif itype == 'exp':
        intval = np.exp(weights[0]*np.log(vals[0])+weights[1]*np.log(vals[1]))
    else:
        raise ValueError('No interpolation type found for int1d')

    return intval


def find_between(array,value,interp='lin'):
    """
    Returns the indexes of the two closest points in array and the relative weights for interpolation.
    :param array: grid Array
    :param value: value requested
    :param interp: type of weighting ('lin' -> linear interpolation,'exp' -> exponential interpolation,
    'box' -> nearest neighbour,'1cos' -> 1/cos(x) for SZA values)
    :return:
    """
    ndim = array.ndim
    dists = np.sort(np.abs(array-value))
    indxs = np.argsort(np.abs(array-value))
    vals = array[indxs][0:2]
    idx = indxs[0:2]
    dis = dists[0:2]
    weights = np.array(weight(value,vals[0],vals[1],itype=interp))
    return idx, weights


def weight(p,pg1,pg2,itype='lin'):
    """
    Weight of pg1 and pg2 in calculating the interpolated value p, according to type.
    :param pg1: Point 1 in the grig
    :param pg2: Point 2 in the grid
    :param itype: 'lin' : (1-difference)/step; 'exp' : (1-log difference)/log step; 'box' : 1 closer, 0 the other; '1cos' : (1-1cos_diff)/1cos_step
    :return:
    """
    w1 = 0
    w2 = 0
    if itype == 'lin':
        w1 = 1-abs(p-pg1)/abs(pg2-pg1)
        w2 = 1-abs(p-pg2)/abs(pg2-pg1)
    elif itype == 'exp':
        pg1log = np.log(pg1)
        pg2log = np.log(pg2)
        plog = np.log(p)
        w1 = 1-abs(plog-pg1log)/abs(pg2log-pg1log)
        w2 = 1-abs(plog-pg2log)/abs(pg2log-pg1log)
    elif itype == 'box':
        if abs(p-pg1) < abs(p-pg2):
            w1=1
            w2=0
        else:
            w1=0
            w2=1
    elif itype == '1cos':
        pg11cos = 1/np.cos(rad(pg1))
        pg21cos = 1/np.cos(rad(pg2))
        p1cos = 1/np.cos(rad(p))
        w1 = 1-abs(p1cos-pg11cos)/abs(pg21cos-pg11cos)
        w2 = 1-abs(p1cos-pg21cos)/abs(pg21cos-pg11cos)

    return w1, w2


def dist(p1,p2):
    """
    Distance between two points in ndim.
    :param p1: Point 1
    :param p2: Point 2
    :return:
    """
    dist = 0
    for e1,e2 in zip(p1,p2):
        dist += (e1-e2)**2
    dist = mt.sqrt(dist)

    return dist



#####################################################################################################################
#####################################################################################################################
###############################             FUNZIONI                 ################################################
#####################################################################################################################
#####################################################################################################################


def integr_sol(wl, spe, wl_range = None, sol_lim = None):
    """
    Integrates the signal in the wl range considered, subtracting a linear solar contribution: the solar contribution is calculated averaging the LINEARLY interpolating between two average values calculated at the left (sol_lim[0][0] < wl < sol_lim[0][1]) and at the right (sol_lim[1][0] < wl < sol_lim[1][1]) of the range considered.
    """
    if wl_range is None:
        wl_range = [np.min(wl),np.max(wl)]

    cond = (wl > wl_range[0]) & (wl < wl_range[1]) & (~np.isnan(spe))
    if sol_lim is not None:
        p1_cond = (wl > sol_lim[0][0]) & (wl < sol_lim[0][1])
        p2_cond = (wl > sol_lim[1][0]) & (wl < sol_lim[1][1])
        wl1 = np.mean(wl[p1_cond])
        wl2 = np.mean(wl[p2_cond])
        sp1 = np.nanmean(spe[p1_cond])
        sp2 = np.nanmean(spe[p2_cond])
        sol = lambda x: sp1+(x-wl1)/(wl2-wl1)*(sp2-sp1)
    else:
        sol = lambda x: 0.0

    fondo = np.array([sol(wlu) for wlu in wl[cond]])
    fondo[np.isnan(fondo)] = 0.0
    intt = np.trapz(spe[cond]-fondo,x=wl[cond])

    return intt


def cbar_things(levels):
    log2 = int(mt.ceil(mt.log10(np.max(levels)))-1)
    log1 = int(mt.ceil(mt.log10(np.min(levels)))-1)

    expo = log2
    if(log1 < log2-1): print('from cbar_things -> Maybe better in log scale?\n')

    if expo == 0 or expo == 1 or expo == -1 or expo == 2:
        lab = ''
        expo = 0
    else:
        lab = r'$\times 10^{{{}}}$ '.format(expo)

    print(log1,log2,expo,np.max(levels),np.min(levels))
    return expo, lab


def map_contour(nomefile, x, y, quant, continuum = True, lines = True, levels=None, ncont=12, cbarlabel='quant', xlabel='x', ylabel='y', ylim=None, xlim=None, cbarform = '%.1f', live = False):
    """
    Makes lat/alt, lat/time or whichever type of 2D contour maps.
    :param x: X coordinate (n x m matrix)
    :param y: Y coordinate (n x m matrix)
    :param quant: Quantity to be plotted (n x m matrix), MaskedArray
    :param contype: If 'continuum_with_levels', color contourf with almost continous shading, with contour lines at the levels. If 'continuum', no contour lines are plotted. If 'discrete' or 'discrete_with_levels' the contourf is done on the ncont levels.
    :param levels: If set levels are fixed by user input.
    :param ncont: Number of levels. If levels is set, ncont is ignored.
    :param cbarlabel: Label for the colorbar. Should contain a {} for exponential in the units, if needed.
    """

    if type(quant) is not np.ma.core.MaskedArray:
        conan = np.isnan(quant)
        quant = np.ma.MaskedArray(quant, conan)

    if live:
        pl.ion()

    fig = pl.figure(figsize=(8, 6), dpi=150)

    #pl.grid()
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    if ylim is not None:
        pl.ylim(ylim[0],ylim[1])
    if xlim is not None:
        pl.xlim(xlim[0],xlim[1])

    if levels is None:
        levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)

    if continuum:
        clevels = np.linspace(levels[0],levels[-1],100)
        pre = np.linspace(np.min(quant.compressed()), levels[0], 10)
        post = np.linspace(levels[-1], np.max(quant.compressed()), 10)
        clevels = np.append(pre[:-1], clevels)
        clevels = np.append(clevels, post[1:])
    else:
        clevels = levels

    print('levels: ',levels)
    print('clevels: ',clevels)
    # fig2 = pl.figure(figsize=(8, 6), dpi=150)
    # pl.plot(np.arange(len(clevels)),clevels)
    # pl.scatter(np.arange(len(clevels)),clevels)
    # pl.show()

    expo, clab = cbar_things(levels)
    quant = quant/10**expo
    levels = levels/10**expo
    clevels = clevels/10**expo
    print(levels)

    zuf = pl.contourf(x,y,quant,corner_mask = True,levels = clevels, linewidths = 0., extend = 'both')
    if lines:
        zol = pl.contour(x,y,quant,levels = levels, colors = 'grey', linewidths = 2.)

    # This is the fix for the white lines between contour levels
    for coz in zuf.collections:
        coz.set_edgecolor("face")
    cb = pl.colorbar(mappable = zuf, format=cbarform, pad = 0.1)
    cb.set_label(cbarlabel.format(clab))

    lol = nomefile.find('.')
    form = nomefile[lol+1:]
    fig.savefig(nomefile, format=form, dpi=150)

    pl.close(fig)

    return


def interpNd(arr,grid,point):
    """
    Interpolates
    :param arr:
    :param point:
    :return:
    """


def trova_spip(file, hasha = '#', read_past = False):
    """
    Trova il '#' nei file .dat
    """
    gigi = 'a'
    while gigi != hasha :
        linea = file.readline()
        gigi = linea[0]
    else:
        if read_past:
            return linea[1:]
        else:
            return

def find_spip(*args, **kwargs):
    out = trova_spip(*args, **kwargs)
    return out

def read_obs(filename):
    """
    Reads files of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    line = infile.readline()
    cosi = line.split()
    n_freq = int(cosi[0])
    n_limb = int(cosi[1])
    trova_spip(infile)
    dists = []
    while len(dists) < n_limb:
        line = infile.readline()
        dists += list(map(float, line.split()))
    dists = np.array(dists)
    trova_spip(infile)
    alts = []
    while len(alts) < n_limb:
        line = infile.readline()
        alts += list(map(float, line.split()))
    alts = np.array(alts)
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = [float(r) for r in data_arr[:, 0]]
    obs = data_arr[:, 1:2*n_limb+2:2]
    obs = obs.astype(np.float)
    flags = data_arr[:, 2:2*n_limb+2:2]
    flags = flags.astype(np.int)
    infile.close()
    return n_freq, n_limb, dists, alts, freq, obs, flags


def writevec(file,vec,n_per_line,format_str):
    """
    Writes a vector in formatted output to a file.
    :param file: File to write to.
    :param vec: Vector to be written
    :param n_per_line: Number of elements of vector per line
    :param format_str: String format of each number written
    :return: nada
    """
    n = len(vec)
    com = n/n_per_line
    for i in range(com):
        i1 = i*n_per_line
        i2 = i1 + n_per_line
        strin = n_per_line*format_str+'\n'
        file.write(strin.format(*vec[i1:i2]))
    nres = n - com * n_per_line
    i1 = com * n_per_line
    if(nres > 0):
        strin = nres*format_str+'\n'
        file.write(strin.format(*vec[i1:n]))

    return


def write_obs(n_freq, n_limb, dists, alts, freq, obs, flags, filename, old_file = 'None'):
    """
    Writes files of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    from datetime import datetime

    infile = open(filename, 'w')
    data = datetime.now()
    infile.write('Modified on: {}\n'.format(data))
    infile.write('Original file: {}\n'.format(old_file))
    infile.write('\n')
    infile.write('Number of spectral points, number of tangent altitudes:\n')
    infile.write('{:1s}\n'.format('#'))
    infile.write('{:12d}{:12d}\n'.format(n_freq,n_limb))
    infile.write('\n')
    infile.write('Altitudes of satellite (km):\n')
    infile.write('{:1s}\n'.format('#'))
    writevec(infile,dists,8,'{:15.4e}')
    #infile.write((8*'{:15.4e}'+'\n').format(dists))
    infile.write('\n')
    infile.write('Tangent altitudes (km): \n')
    infile.write('{:1s}\n'.format('#'))
    writevec(infile,alts,8,'{:10.2f}')
    #infile.write((8*'{:10.2f}'+'\n').format(alts))
    infile.write('\n')
    infile.write('Wavelength (nm), spectral data (W m^-2 nm^-1 sr^-1):\n')
    infile.write('{:1s}\n'.format('#'))

    for fr, ob, fl in zip(freq, obs, flags):
        str = '{:10.4f}'.format(fr)
        for oo, ff in zip(ob,fl):
            str = str + '{:15.4e}{:3d}'.format(oo,ff)
        str = str + '\n'
        infile.write(str)

    infile.close()
    return


def read_input_prof_gbb(filename, type, n_alt = 151, alt_step = 10.0, n_gas = 86, n_lat = 4):
    """
    Reads input profiles from gbb standard formatted files (in_temp.dat, in_pres.dat, in_vmr_prof.dat).
    Profile order is from surface to TOA.
    type = 'vmr', 'temp', 'pres'
    :return: profiles
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'r')

    if(type == 'vmr'):
        print(type)
        trova_spip(infile)
        trova_spip(infile)
        first = 1
        for i in range(n_gas):
            lin = infile.readline()
            #print(lin)
            num = lin.split()[0]
            nome = lin.split()[1]
            prof = []
            while len(prof) < n_alt:
                line = infile.readline()
                prof += list(map(float, line.split()))
            prof = np.array(prof[::-1])
            if(first):
                proftot = prof
                first = 0
            else:
                proftot = np.vstack([proftot,prof])
            for j in range(n_lat-1): # to skip other latitudes
                prof = []
                while len(prof) < n_alt:
                    line = infile.readline()
                    prof += list(map(float, line.split()))
        proftot = np.array(proftot)


    if(type == 'temp' or type == 'pres'):
        print(type)
        trova_spip(infile)
        trova_spip(infile)
        prof = []
        while len(prof) < n_alt:
            line = infile.readline()
            prof += list(map(float, line.split()))
        proftot = np.array(prof[::-1])

    return proftot


def write_input_prof_gbb(filename, prof, type, n_alt = 151, alt_step = 10.0, nlat = 4, descr = '', script=__file__):
    """
    Writes input profiles in gbb standard formatted files (in_temp.dat, in_pres.dat, in_vmr_prof.dat)
    Works both with normal vectors and with sbm.AtmProfile objects.
    :return:
    """
    from datetime import datetime

    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'w')
    data = datetime.now()
    infile.write(descr+'\n')
    infile.write('\n')
    infile.write('Processed through -{}- on: {}\n'.format(script,data))
    infile.write('{:1s}\n'.format('#'))

    n_per_line = 8

    if(type == 'vmr'):
        strin = '{:10.3e}'
        infile.write('VMR of molecules (ppmV)\n')
    elif(type == 'temp'):
        strin = '{:10.5f}'
        infile.write('Temperature (K)\n')
    elif(type == 'pres'):
        strin = '{:11.4e}'
        infile.write('Pressure (hPa)\n')
    else:
        raise ValueError('Type not recognized. Should be one among: {}'.format(['vmr','temp','pres']))

    infile.write('{:1s}\n'.format('#'))
    for i in range(nlat):
        writevec(infile,prof[::-1],n_per_line,strin)

    return


def read_tvib_gbb(filename, atmosphere, grid = None, l_ratio = True, n_alt = 151, alt_step = 10.0, nlat = 4):
    """
    Reads in_vibtemp.dat file. Output is a list of sbm.Molec objects. Atmosphere is a sbm.AtmProfile object with at
    """
    if grid is None:
        alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)
    else:
        alts = grid[0]
        n_alt = len(alts)

    temp = atmosphere.interp_copy('temp',alts)
    infile = open(filename, 'r')

    trova_spip(infile)
    n_mol = int(infile.readline().rstrip())

    molecs = []
    mol = 0
    molec = Molec(mol, 'mol_0', MM=None)

    for ii in range(n_mol):
        trova_spip(infile)
        mol, iso, n_lev = map(int, infile.readline().rstrip().split())
        print(mol,iso,n_lev)
        if mol != molec.mol: # Se è una molecola nuova ricreo il molec
            if molec.mol != 0: molecs.append(molec)
            molec = Molec(mol, 'mol_{:2d}'.format(mol), MM=None)
            molec.link_to_atmos(atmosphere)
        molec.add_iso(iso)

        for ll in range(n_lev):
            trova_spip(infile)
            linea = infile.readline().rstrip()
            lev_str = linea[0:15]
            print(linea.split())
            n_lev_int, n_simm = map(int,linea[15:].split()[:2])
            try:
                fac, energy = map(float,linea[15:].split()[2:])
            except:
                fac = float(linea[15:].split()[2])
                energy = find_ch4_energy(lev_str)
            simms = []
            for sss in range(n_simm):
                simms.append(infile.readline().rstrip()[0:15])

            trova_spip(infile)
            prof = []
            while len(prof) < n_alt:
                line = infile.readline()
                prof += list(map(float, line.split()))
            prof = np.array(prof[::-1])
            prof = ratio_to_vibtemp(energy, prof, temp)
            prof = AtmProfile(prof, alts, profname = 'vibtemp')

            # levello = Level(lev_str, energy, degen = -1, simmetry = simms)
            # levello.add_vibtemp(prof)
            isomol = getattr(molec, 'iso_{:1d}'.format(iso))
            isomol.add_level(lev_str, energy, degeneracy = -1, simmetry = simms, vibtemp = prof)
            trova_spip(infile)

    molecs.append(molec)

    return molecs


def find_ch4_energy(lev_string):
    energies_ch4 = [0.0, 1310.7606, 1533.332, 2608.69, 2838.26, 2916.483, 3019.497, 3064.48, 4223.46, 4321.67, 4540.65, 6024.28, 3870.5, 5588.0, 4123.0, 5161.0958, 5775.0, 5861.0]
#    levels = [['     0 0 0 0   ','    0 0 0 0 1A1'], ['     0 0 0 1   ', '    0 0 0 1 1F2'], ['     0 1 0 0   '], ['     0 0 0 2   ', '    0 0 0 2 1F2', '    0 0 0 2 1E', '    0 0 0 2 1A1'], ['     0 1 0 1   ', '    0 1 0 1 1F2', '    0 1 0 1 1F1'], ['     1 0 0 0   ', '    1 0 0 0 1A1'], ['     0 0 1 0   ', '    0 0 1 0 1F2'], ['     0 2 0 0   ', '    0 2 0 0 1E', '    0 2 0 0 1A1', '    0 2 0 0 1 E'], ['     1 0 0 1   ', '    1 0 0 1 1F2'], ['     0 0 1 1   ', '    0 0 1 1 1F1', '    0 0 1 1 1F2', '    0 0 1 1 1E', '    0 0 1 1 1A1'], ['     0 1 1 0   ', '    0 1 1 0 1F1', '    0 1 1 0 1F2'], ['     0 0 2 0   ', '    0 0 2 0 1F2', '    0 0 2 0 1E', '    0 0 2 0 1A1'], ['     0 0 0 3   ', '    0 0 0 3 1F1', '    0 0 0 3 2F2'], ['     0 0 1 2   ', '    0 0 1 2 1F1', '    0 0 1 2 1F2', '    0 0 1 2 1E', '    0 0 1 2 1A1'], ['     0 1 0 2   ', '    0 1 0 2 1F2', '    0 1 0 2 1F1', '    0 1 0 2 1E', '    0 1 0 2 1A2', '    0 1 0 2 2 E', '    0 1 0 2 1A1'], ['     0 0 0 4   '], ['     1 1 0 1   '], ['     0 1 1 1   ', '    0 1 1 1 1F1', '    0 1 1 1 1F2', '    0 1 1 1 1E', '    0 1 1 1 1A1', '    0 1 1 1 1A2']]

    lev_quanta = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 2], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 1, 2], [0, 1, 0, 2], [0, 0, 0, 4], [1, 1, 0, 1], [0, 1, 1, 1]]


    for lista_cosi, energy in zip(lev_quanta,energies_ch4):
        quant = extract_quanta_ch4(lev_string)[0]
        print(quant,lista_cosi)

        if quant == lista_cosi:
            return energy

    raise ValueError('Level {} not found!'.format(lev_string))

def extract_quanta_ch4(lev_string):
    v1, v2, v3, v4 = map(int,lev_string.split()[:4])
    try:
        simm = lev_string.split()[4].strip()
    except:
        simm = ''

    return [v1,v2,v3,v4], simm

def hitran_formatter(quanta,simmetry = '', molec = 'CH4'):

    lista = [q for q in quanta] + [simmetry]

    if molec == 'CH4':
        string = 4*' '+4*'{:1d} '+'{:3s}'
        string = string.format(*lista)

    return string

def read_mol_levels_HITRAN(filename = None, molec = None):
    """
    Reads molecular levels strings from an external file. Format is a bit weird..
    """

    if filename is None and molec is None:
        raise ValueError('Please insert the filename or the molecule to be loaded.\n')
    elif filename is None:
        if molec == 'CH4':
            filename = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/CH4_levels_bianca.dat'

    fil = open(filename,'r')

    n_simms = []
    lev_strings = []
    quanta = []

    find_spip(fil)
    lines = fil.readlines()
    ilin = 0
    while ilin < len(lines):
        line = lines[ilin]
        ilin += 1
        print(line,len(line))
        nome = line[:9]
        #print(line[9:18])
        gigi = extract_quanta_ch4(line[9:18])[0]
        #print(gigi)
        quanta.append(gigi)
        n_simm = int(line[19:25])
        n_simms.append(n_simm)
        lev_stringa_lui = []
        quantsim = extract_quanta_ch4(line[25:39].rstrip())
        lev_stringa_lui.append(hitran_formatter(quantsim[0],quantsim[1],molec='CH4'))
        for isim in range(n_simm-1):
            line = lines[ilin]
            print(line,len(line))
            quantsim = extract_quanta_ch4(line[25:39].rstrip())
            lev_stringa_lui.append(hitran_formatter(quantsim[0],quantsim[1],molec='CH4'))
            ilin += 1

        lev_strings.append(lev_stringa_lui)

    key_strings = ['quanta', 'n_simms', 'lev_strings']
    variables = [quanta,n_simms,lev_strings]

    return dict(zip(key_strings,variables))


def write_tvib_gbb(filename, molecs, atmosphere, grid = None, l_ratio = True, n_alt = 151, alt_step = 10.0, nlat = 4, descr = '', script=__file__):
    """
    Writes input in_vibtemp.dat in gbb standard format: nlte ratio.
    :param molecs: has to be a single sbm.Molec object or a list of sbm.Molec objects
    :param atmosphere: sbm.AtmProfile object.
    :return:
    """
    from datetime import datetime

    # Check dimension of grids for temp and vibtemps
    if grid is None:
        alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)
    else:
        alts = grid[0]
        n_alt = len(alts)

    if n_alt != np.shape(atmosphere.grid)[1]:
        try:
            temp = atmosphere.interp_copy('temp',alts)
        except Exception as cazzillo:
            print(cazzillo)
            print("Using name 'prof' instead")
            temp = atmosphere.interp_copy('prof',alts)
    else:
        try:
            temp = atmosphere.temp
        except Exception as cazzillo:
            print(cazzillo)
            print("Using name 'prof' instead")
            temp = atmosphere.prof

    try:
        n_mol = len(molecs)
        n_iso_tot = np.sum(np.array([mol.n_iso for mol in molecs]))
    except:
        n_mol = 1
        n_iso_tot = molecs.n_iso
        molecs = [molecs]

    infile = open(filename, 'w')
    n_per_line = 8

    if l_ratio:
        strin = '{:10.3e}'
    else:
        strin = '{:10.3f}'

    data = datetime.now()
    infile.write(descr+'\n')
    infile.write('\n')
    if l_ratio:
        infile.write('Type: non-LTE ratios\n')
    else:
        infile.write('Type: vibrational temperatures\n')
    infile.write('\n')
    infile.write('Processed through -{}- on: {}\n'.format(script,data))
    infile.write('\n')
    infile.write('Number of molecules:\n')
    infile.write('{:1s}\n'.format('#'))
    infile.write('{}\n'.format(n_iso_tot))
    infile.write('\n')

    for mol in molecs:
        print(mol,type(mol))
        for iso in mol.all_iso:
            print(iso)
            _iso_ = getattr(mol,iso)
            print(type(_iso_))
            infile.write('Hitran mol number, iso number and number of levels included here:\n')
            infile.write('{:1s}\n'.format('#'))
            infile.write('{:5d}{:5d}{:5d}\n'.format(_iso_.mol,_iso_.iso,_iso_.n_lev))

            ioo=0
            for lev in _iso_.levels:
                print(lev)
                _lev_ = getattr(_iso_, lev)
                print(type(_lev_))
                infile.write('Hitran Code of level, internal code of level, number of simmetries, multiplier(depr.), Level Energy:\n')
                infile.write('{:1s}\n'.format('#'))
                print(_lev_.simmetry)
                infile.write('{:15.15s}{:5d}{:5d}{:20.3f}{:12.4f}\n'.format(_lev_.simmetry[0],ioo,len(_lev_.simmetry)-1,1.0,_lev_.energy))
                if len(_lev_.lev_string) != 15:
                    errstring = 'Length of level string is {} instead of 15 for level {} of molecule {}.'.format(len(_lev_.lev_string),lev,iso)
                    raise ValueError(errstring)

                for simm in _lev_.simmetry[1:]:
                    infile.write('{:15s}\n'.format(simm))

                # CHECKS if vibtemp profile has the right dimension:
                if n_alt != np.shape(_lev_.vibtemp.grid)[1]:
                    try:
                        vibtemp = _lev_.vibtemp.interp_copy('vibtemp',alts)
                    except Exception as cazzillo:
                        print(cazzillo)
                        print("Using name 'prof' instead")
                        vibtemp = _lev_.vibtemp.interp_copy('prof',alts)
                else:
                    try:
                        vibtemp = _lev_.vibtemp.temp
                    except Exception as cazzillo:
                        print(cazzillo)
                        print("Using name 'prof' instead")
                        vibtemp = _lev_.vibtemp.prof


                infile.write('{:1s}\n'.format('#'))
                if l_ratio:
                    print('Energiaaa ',_lev_.energy)
                    prof = vibtemp_to_ratio(_lev_.energy, vibtemp, temp)
                    #prof = np.exp(-_lev_.energy/kbc*(1/vibtemp-1/temp))
                else:
                    prof = vibtemp

                for i in range(nlat):
                    writevec(infile,prof[::-1],n_per_line,strin)
                infile.write('{:1s}\n'.format('#'))
                ioo+=1

    infile.close()

    return


def read_input_atm_man(filename):
    """
    Reads input atmosphere in manuel standard.
    :param filename:
    :return:
    """
    infile = open(filename,'r')
    trova_spip(infile,hasha='$')
    n_alt = int(infile.readline())
    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    alts = np.array(prof)

    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    pres = np.array(prof)

    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    temp = np.array(prof)

    return alts, temp, pres


def read_tvib_manuel(filename):
    """
    Reads input atmosphere in manuel standard.
    :param filename:
    :return:
    """
    infile = open(filename,'r')
    trova_spip(infile,hasha='$')
    n_alt = int(infile.readline())
    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    alts = np.array(prof)

    trova_spip(infile,hasha='$')
    l_vib = infile.readline().rstrip()
    if l_vib == 'F':
        print('Reading VIBRATIONAL TEMPERATURES')
    elif l_vib == 'T':
        print('Reading POPULATION RATIOS')

    trova_spip(infile,hasha='$')
    n_lev = int(infile.readline())

    molecs = []
    levels = []
    energies = []
    vibtemps = []
    for i in range(n_lev):
        linea = trova_spip(infile,hasha = '$',read_past = True)
        levels.append(linea[0:15])
        energies.append(float(linea[15:]))
        molecs.append(infile.readline().rstrip())
        infile.readline()

        prof = []
        while len(prof) < n_alt:
            line = infile.readline()
            prof += list(map(float, line.split()))
        pres = np.array(prof)
        vibtemps.append(pres)

    infile.close()

    return alts, molecs, levels, energies, vibtemps


def write_input_atm_man(filename, z, T, P, n_alt = 301, alt_step = 5.0):
    """
    Writes input profiles in manuel standard formatted files
    :return:
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'w')
    n_per_line = 5


    str1 = '{:8.1f}'
    str2 = '{:11.4e}'
    str3 = '{:9.3f}'

    infile.write('# Atmosphere with wavy prof, reference Atm 05 S, 2006/07\n')
    infile.write('\n')
    infile.write('Number of levels\n')
    infile.write('{:1s}\n'.format('$'))
    infile.write('{}\n'.format(n_alt))
    infile.write('\n')
    infile.write('Altitudes [km]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,z,n_per_line,str1)
    infile.write('\n')
    infile.write('Pressure [hPa]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,P,n_per_line,str2)
    infile.write('\n')
    infile.write('Temperature [K]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,T,n_per_line,str3)
    infile.close()

    return


def read_input_prof_lin(filename, n_col, n_alt = 151, alt_step = 10.0):
    """
    Reads input profiles from
    :return:
    """


def write_input_prof_lin():
    """
    Reads input profiles from gbb
    :return:
    """


def scriviinputmanuel(alts,temp,pres,filename):
    """
    Writes input PT for fomichev in Manuel's format.
    :return:
    """
    fi = open(filename,'w')
    fi.write('Number of levels\n')
    fi.write('{:1s}\n'.format('$'))
    fi.write('{:5d}\n'.format(len(alts)))

    fi.write('\n')
    fi.write('altitude (km)\n')
    fi.write('{:1s}\n'.format('$'))
    writevec(fi,alts,5,'{:11.1f}')

    fi.write('\n')
    fi.write('pressure (hPa)\n')
    fi.write('{:1s}\n'.format('$'))
    writevec(fi,pres,8,'{:11.3e}')

    fi.write('\n')
    fi.write('temperature (K)\n')
    fi.write('{:1s}\n'.format('$'))
    writevec(fi,temp,8,'{:11.3e}')

    fi.close()
    return


def leggioutfomi(nomeout):
    """
    Reads Fomichev output.
    :param nomeout:
    :return:
    """
    fi = open(nomeout,'r')
    trova_spip(fi)

    data = np.array([map(float, line.split()) for line in fi])

    alt_fomi = np.array(data[:,0])
    cr_fomi = np.array(data[:,5])

    return alt_fomi, cr_fomi


def read_sim_gbb(filename,skip_first = 0, skip_last = 0):
    """
    Read sim_*.dat or spet_*.dat files in gbb format.
    :return:
    """
    infile = open(filename, 'r')
    line = infile.readline()
    line = infile.readline()
    alt = line.split()[0]
    trova_spip(infile)

    for i in range(skip_first):
        line = infile.readline()

    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = np.array([float(r) for r in data_arr[:, 1]])
    obs = np.array([float(r) for r in data_arr[:, 2]])
    sim = np.array([float(r) for r in data_arr[:, 3]])
    err = np.array([float(r) for r in data_arr[:, 4]])
    #flags = data_arr[:, 2:2*n_limb+2:2]
    #flags = flags.astype(np.int)
    infile.close()

    if skip_last == 0:
        return alt,freq,obs,sim,err
    else:
        return alt,freq[:-skip_last],obs[:-skip_last],sim[:-skip_last],err[:-skip_last]


def HG_phase_funct(deg,g):
    """
    Henyey-Greenstein phase function.
    :param deg: angle in radiants
    :param g: asymmetry factor (from 0 to 1)
    :return:
    """
    phunc=(1.0/(4.0*mt.pi))*(1.0-g**2)/(1.0+g**2+2.0*g*mt.cos(deg))**1.5

    return phunc


def read_inputs(nomefile, key_strings, n_lines = None, itype = None, defaults = None, verbose=False):
    """
    Standard reading for input files. Searches for the keys in the input file and assigns the value to variable.
    :param keys: List of strings to be searched in the input file.
    :param defaults: List of default values for the variables.
    :param n_lines: List. Number of lines to be read after key.
    """

    keys = ['['+key+']' for key in key_strings]

    if n_lines is None:
        n_lines = np.ones(len(keys))

    if itype is None:
        itype = len(keys)*[None]

    if defaults is None:
        warnings.warn('No defaults are set. Setting None as default value.')
        defaults = len(keys)*[None]

    variables = []
    is_defaults = []
    with open(nomefile, 'r') as infile:
        lines = infile.readlines()
        for key, deflt, nli, typ in zip(keys,defaults,n_lines,itype):

            is_key = np.array([key in line for line in lines])
            if np.sum(is_key) == 0:
                print('Key {} not found, setting default value {}\n'.format(key,deflt))
                variables.append(deflt)
                is_defaults.append(True)
            elif np.sum(is_key) > 1:
                raise KeyError('Key {} appears {} times, should appear only once.'.format(key,np.sum(is_key)))
            else:
                num_0 = np.argwhere(is_key)[0][0]
                if nli == 1:
                    cose = lines[num_0+1].split()
                    if typ == bool: cose = [str_to_bool(lines[num_0+1].split()[0])]
                    if typ == str: cose = [lines[num_0+1].rstrip()]
                    if len(cose) == 1:
                        variables.append(map(typ,cose)[0])
                    else:
                        variables.append(map(typ,cose))
                else:
                    cose = []
                    for li in range(nli):
                        cos = lines[num_0+1+li].split()
                        if typ == str: cos = [lines[num_0+1+li].rstrip()]
                        if len(cos) == 1:
                            cose.append(map(typ,cos)[0])
                        else:
                            cose.append(map(typ,cos))
                    variables.append(cose)
                is_defaults.append(False)

    if verbose:
        for key, var, deflt in zip(keys,variables,is_defaults):
            print('----------------------------------------------\n')
            if deflt:
                print('Key: {} ---> Default Value: {}\n'.format(key,var))
            else:
                print('Key: {} ---> Value Read: {}\n'.format(key,var))

    return dict(zip(key_strings,variables))


def str_to_bool(s):
    if s == 'True' or s == 'T':
         return True
    elif s == 'False' or s == 'F':
         return False
    else:
         raise ValueError('Not a boolean value')


def read_rannou_aer(filename):
    """
    Reads the aerosol properties (ext. coeff., ssa and the 256 coefficients of Legendre polinomials for the phase function)
    :return: freq (nm), extcoeff (cm-1), ssa, matrix(n_freq,n_poli)
    """
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = np.array([float(r) for r in data_arr[:, 0]])
    extco = np.array([float(r) for r in data_arr[:, 1]])
    ssa = np.array([float(r) for r in data_arr[:, 2]])
    leg_coeff = data_arr[:, 3:]
    leg_coeff = leg_coeff.astype(np.float)
    infile.close()

    return freq,extco,ssa,leg_coeff


def gaussian(x, mu, fwhm):
    pi = mt.acos(-1)
    sig = fwhm / (2*mt.sqrt(2*mt.log(2.0)))
    fac = mt.sqrt(2*pi)*sig
    return np.exp(-(x - mu)**2 / (2 * sig**2)) / fac


def freqTOwl(freq,spe_freq,wl,fwhm):
    """
    Converts a HIRES spectrum in nW/(cm2*cm-1) into a LOWRES spectrum in W/(m2*nm)
    :param freq: HIRES freq (cm-1)
    :param spe_freq: HIRES spectrum ( nW/(cm2*cm-1) )
    :param wl: LOWRES wl grid (nm)
    :param fwhm: FWMH of the ILS (nm)
    :return: spe : LOWRES spectrum in W/(m2*nm)
    """

    spe_freq = 10**(-5)*freq**2/10**7 * spe_freq
    freq = 10**7/freq

    freq = freq[::-1] # reordering freq
    spe_freq = spe_freq[::-1]

    spe = []
    for w, fw in zip(wl, fwhm):
        gauss = gaussian(freq, w, fw)
        convol = np.trapz(gauss*spe_freq, x=freq)
        spe.append(float(convol))

    spe = np.array(spe)

    return spe


def read_bands(filename):
    """
    Reads bands and ILS fwhm of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    wl = [float(r) for r in data_arr[:, 0]]
    fwhm = [float(r) for r in data_arr[:, 2]]
    infile.close()

    return wl, fwhm


def findcol(n,i, color_map = 5):
    """
    Gives the best COLOR choice for line i in a n-lines plot.
    :param n: total number of lines
    :param i: line considered
    :param color_map: OPTIONAL, name of the color map.
    :return: RGBA tuple like (0.1,0.4,0.3,1.0)
    """
    import matplotlib.cm as cm

    cmaps = ['spectral','jet','gist_ncar','gist_rainbow','hsv','nipy_spectral']

    if n < 5:
        oss = 0.2
        fa = 0.6
    elif n > 5 and n < 11:
        oss = 0.1
        fa = 0.8
    else:
        oss = 0.0
        fa = 1.0

    cmap = cm.get_cmap(cmaps[color_map])
    colo = oss+fa*i/(n-1)

    # setting linestyle
    lis = ['--','-','-.','-']#,':']
    oi = i % 4

    return cmap(colo), lis[oi]


def plotta_sim_VIMS(nomefile,freq,obs,sim,sims,names,err=1.5e-8,title='Plot', auto = True,
                    xscale=[-1,-1],yscale=[-1,-1],yscale_res=[-1,-1]):
    """
    Plots observed/simulated with residuals and single contributions.
    :param obs: Observed
    :param sim: Simulated total
    :param n_sims: Number of simulated mol. contributions
    :param sims: matrix with one sim per row
    :param names: names of each sim
    :return:
    """
    from matplotlib.font_manager import FontProperties
    import matplotlib.gridspec as gridspec

    fontP = FontProperties()
    fontP.set_size('small')
#    legend([plot1], "title", )
    n_sims = sims.shape[0]
    fig = pl.figure(figsize=(8, 6), dpi=150)

    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    ax1 = pl.subplot(gs[0])
    pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
    pl.title(title)
    ax2 = pl.subplot(gs[1])
    if not auto:
        ax1.set_xlim(xscale)
        ax1.set_ylim(yscale)
        ax2.set_xlim(xscale)
        ax2.set_ylim(yscale_res)

    pl.xlabel('Wavelength (nm)')

#    pl.subplot(211)
    colo, li = findcol(n_sims+2,-1)
    ax1.plot(freq,obs,color=colo,label='Data',linewidth=1.0)
    ax1.scatter(freq,obs,color=colo,linewidth=1.0)
    ax1.errorbar(freq,obs,color=colo,yerr=err, linewidth=1.0)
    colo, li = findcol(n_sims+2,0)
    ax1.plot(freq,sim,color=colo,linewidth=3.0,label='Sim')
    i=1
    for name,simu in zip(names,sims):
        colo, li = findcol(n_sims+2,i)
        ax1.plot(freq,simu,color=colo,linestyle=li,label=name,linewidth=2.0)
        i +=1
    ax1.grid()
    ax1.legend(loc=1,bbox_to_anchor=(1.05,1.1),fontsize='small',fancybox=1,shadow=1)

#    pl.subplot(212)
    ax2.grid()
    ax2.plot(freq,obs-sim,color='red',linewidth=3.0)
#    ax2.fill_between(freq,err*np.ones(len(freq)),-err*np.ones(len(freq)), facecolor=findcol(12,8)[0], alpha=0.1)
    ax2.plot(freq,err*np.ones(len(freq)),color='black',linestyle='--',linewidth=2.0)
    ax2.plot(freq,-err*np.ones(len(freq)),color='black',linestyle='--',linewidth=2.0)

    fig.savefig(nomefile, format='eps', dpi=150)
    pl.close(fig)

    return


def plot_spect_sim(nomefile,freq,obs,sims,names,err=1.5e-8,title='',
                    xscale=None,yscale=None,yscale_res=None):
    """
    Plots observed/simulated with residuals and single contributions.
    :param obs: Observed
    :param sim: Simulated total
    :param n_sims: Number of simulated mol. contributions
    :param sims: matrix with one sim per row
    :param names: names of each sim
    :return:
    """
    from matplotlib.font_manager import FontProperties
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FormatStrFormatter

    fontP = FontProperties()
    fontP.set_size('small')
#    legend([plot1], "title", )
    n_sims = len(sims)
    fig = pl.figure(figsize=(8, 6), dpi=150)

    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    ax1 = pl.subplot(gs[0])
    pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
    pl.title(title)
    ax2 = pl.subplot(gs[1])
    if xscale is not None:
        ax1.set_xlim(xscale)
        ax2.set_xlim(xscale)
    if yscale is not None:
        ax1.set_ylim(yscale)
        ax2.set_ylim(yscale_res)

    pl.xlabel('Wavelength (nm)')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))

#    pl.subplot(211)
    colo, li = findcol(n_sims+2,-1)
    ax1.plot(freq,obs,color=colo,label='Data',linewidth=1.0)
    ax1.scatter(freq,obs,color=colo,linewidth=1.0)
    ax1.errorbar(freq,obs,color=colo,yerr=err, linewidth=1.0)
    colo, li = findcol(n_sims+2,0)
    # ax1.plot(freq,sim,color=colo,linewidth=3.0,label='Sim')
    i=1
    for name,simu in zip(names,sims):
        colo, li = findcol(n_sims+2,i)
        ax1.plot(freq,simu,color=colo,linestyle=li,label=name,linewidth=2.0)
        i +=1
    ax1.grid()
    ax1.legend(loc=1,bbox_to_anchor=(1.05,1.1),fontsize='small',fancybox=1,shadow=1)

#    pl.subplot(212)
    ax2.grid()
    i=1
    for name,simu in zip(names,sims):
        colo, li = findcol(n_sims+2,i)
        ax2.plot(freq,obs-simu,color=colo,linestyle=li,linewidth=2.0)
        i +=1
    # ax2.plot(freq,obs-sim,color='red',linewidth=3.0)
#    ax2.fill_between(freq,err*np.ones(len(freq)),-err*np.ones(len(freq)), facecolor=findcol(12,8)[0], alpha=0.1)
    ax2.plot(freq,err*np.ones(len(freq)),color='black',linestyle='--',linewidth=1.0)
    ax2.plot(freq,-err*np.ones(len(freq)),color='black',linestyle='--',linewidth=1.0)

    io = nomefile.find('.')
    form = nomefile[io+1:]
    fig.savefig(nomefile, format=form, dpi=150)
    pl.close(fig)

    return


def leggi_der_gbb(filename):
    """
    Reads jacobians in gbb format.
    """
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = [float(r) for r in data_arr[:, 1]]
    ders = data_arr[:, 2:n_sint+2]
    ders = ders.astype(np.float)
    infile.close()
    return freq, ders


def convert_fortformat_to_dtype(formato):
    if formato[0] == '(' and formato[-1] == ')':
        formato = formato[1:-1]

    forms = formato.split(',')
    pass


def read_tabellone_emi(filename):
    """
    Legge i tabelloni geometrici formato emi.
    (A40,3X,I2,3X,I2,3X,A12,3X,A21,3X,
    E13.6,3X,
    F8.3,3X,
    F8.3,3X,
    F8.3,3X,
    F8.3,3X,
    E12.3,3X,
    F8.2,3X,
    A20,3X,A20,3X,A20)
    """

    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)

    names = 'obsid sample line tint res tgalt tglat tglon tgsza tglt dist azim start targname targdesc'
    names = names.split()

    # tg_alts = np.array(map(float,[pix['tgalt'] for pix in pix_tot]))
    # tg_lats = np.array(map(float,[pix['tglat'] for pix in pix_tot]))
    # tg_szas = np.array(map(float,[pix['tgsza'] for pix in pix_tot]))
    # dists = np.array(map(float,[pix['dist'] for pix in pix_tot]))
    # tint = np.array(map(float,[pix['tint'][0] for pix in pix_tot]))
    tg_alts = []
    tg_lats = []
    tg_szas = []
    dists = []
    tint = []
    cubes = []

    pixels = []
    for linea in data_arr:
        #lineuz = np.array(linea,dtype='|S40,i2,i2,|S12,|S21'+7*',f8'+3*',|S20')
        #print(len(linea))
        pix = dict(zip(names,linea))
        try:
            gi = float(pix['tgalt'])
            gi = float(pix['tglat'])
            gi = float(pix['tgsza'])
            gi = float(pix['dist'])
            gitint = float(pix['tint'][1:-1].split(',')[0])
            gi = pix['obsid']
            tg_alts.append(float(pix['tgalt']))
            tg_lats.append(float(pix['tglat']))
            tg_szas.append(float(pix['tgsza']))
            dists.append(float(pix['dist']))
            tint.append(gitint)
            cubes.append(pix['obsid'])
        except Exception as cazzillo:
            # print(linea)
            # print(pix)
            # raise cazzillo
            #print(cazzillo)
            continue
        pixels.append(pix)

    cose = [tg_alts,tg_lats,tg_szas,dists,tint,cubes]

    return pixels, cose


def prova_cmap(cma=None):
    """
    Shows colorbar with selected cmap.
    """

    if cma is not None and cma not in cm.cmap_d.keys():
        print('Name {} not recognized \n'.format(cma))
        cma = None

    if cma is None:
        print('------ Choose color map:\n ------')
        for key in sorted(cm.cmap_d.keys()):
            print(key[0:])
        while cma is None:
            print('------ Enter color map name:\n ------')
            try:
                cma = input()
            except:
                print('Name not recognized (virgolette!)\n'.format(cma))
                pass

    sca = pl.cm.ScalarMappable(cmap=cma)
    sca.set_array(np.linspace(0,1,100))
    pl.colorbar(sca)
    pl.title(cma)
    pl.show()

    return


def plotcorr(x, y, filename, xlabel = 'x', ylabel = 'y', xlim = None, ylim = None, format = 'eps'):
    """
    Plots correlation graph between x and y, fitting a line and calculating Pearson's R coeff.
    :param filename: abs. path of the graph
    :params xlabel, ylabel: labels for x and y axes
    """
    pearR = np.corrcoef(x,y)[1,0]
    A = np.vstack([x,np.ones(len(x))]).T  # A = [x.T|1.T] dove 1 = [1,1,1,1,1,..]
    m,c = np.linalg.lstsq(A,y)[0]
    xlin = np.linspace(min(x)-0.05*(max(x)-min(x)),max(x)+0.05*(max(x)-min(x)),11)

    fig = pl.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.grid()
    pl.scatter(x, y, label='Data', color='blue', s=4, zorder=3)
    if xlim is not None:
        if np.isnan(xlim[1]):
            pl.xlim(xlim[0],pl.xlim()[1])
        elif np.isnan(xlim[0]):
            pl.xlim(pl.xlim()[0],xlim[1])
        else:
            pl.xlim(xlim[0],xlim[1])
    if ylim is not None:
        if np.isnan(ylim[1]):
            pl.ylim(ylim[0],pl.ylim()[1])
        elif np.isnan(ylim[0]):
            pl.ylim(pl.ylim()[0],ylim[1])
        else:
            pl.ylim(ylim[0],ylim[1])
    pl.plot(xlin, xlin*m+c, color='red', label='y = {:8.2f} x + {:8.2f}'.format(m,c))
    pl.title("Pearson's R = {:5.2f}".format(pearR))
    pl.legend(loc=4,fancybox =1)
    fig.savefig(filename, format=format, dpi=150)
    pl.close()

    return


#################################################################################
##                                                                            ###
##                                                                            ###
##       GEOMETRICAL FUNCTIONS relating with orbit, atmospheric grid, ...    ####
##                                                                            ###
##                                                                            ###
#################################################################################



def szafromsspDEG(lat, lon, lat_ss, lon_ss):
    """
    Returns sza at certain (lat, lon) given (lat_ss, lon_ss) of the sub_solar_point.
    All angles in DEG.
    """
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0
    lat_ss = lat_ss *np.pi/180.0
    lon_ss = lon_ss *np.pi/180.0
    sc_prod = mt.cos(lat) * mt.cos(lat_ss) * mt.cos(lon) * mt.cos(lon_ss) \
              + mt.cos(lat) * mt.cos(lat_ss) * mt.sin(lon) * mt.sin(lon_ss) \
              + mt.sin(lat) * mt.sin(lat_ss)
    sza = mt.acos(sc_prod) * 180.0 / np.pi
    return sza


def szafromssp(lat, lon, lat_ss, lon_ss):
    """
    Returns sza at certain (lat, lon) given (lat_ss, lon_ss) of the sub_solar_point.
    All angles in radians.
    """
    sc_prod = mt.cos(lat) * mt.cos(lat_ss) * mt.cos(lon) * mt.cos(lon_ss) \
              + mt.cos(lat) * mt.cos(lat_ss) * mt.sin(lon) * mt.sin(lon_ss) \
              + mt.sin(lat) * mt.sin(lat_ss)
    sza = mt.acos(sc_prod)
    return sza


def rad(ang):
    return ang*np.pi/180.0


def deg(ang):
    return ang*180.0/np.pi


def sphtocart(r, R=Rtit, input_in_deg = True):
    """
    Converts from (lat, lon, alt) to (x, y, z).
    Convention: lon goes from 0 to 360, starting from x axis, towards East. lat is the latitude.
    h is the altitude with respect to the spherical surface. R_p is the planet radius.
    :return: 3D numpy array
    """
    lat = r[0]
    lon = r[1]
    h = r[2]

    if input_in_deg:
        latu = rad(lat)
        lonu = rad(lon)

    r = [mt.cos(latu)*mt.cos(lonu), mt.cos(latu)*mt.sin(lonu), mt.sin(latu)]
    r = np.array(r)
    r *= (R + h)

    return r


def carttosph(r, R = Rtit, output_in_deg = True):
    """
    Converts to (lat, lon, alt) from (x, y, z).
    Convention: lon goes from 0 to 360, starting from x axis, towards East. lat is the latitude.
    h is the altitude with respect to the spherical surface. R_p is the planet radius.
    :return: 3D numpy array
    """
    dist = LA.norm(r)
    h = dist - R
    lat = mt.asin(r[2]/dist)
    #lon = mt.atan(r[1]/r[0])
    #print('eja',r[0]/(dist*mt.cos(lat)),r[0]/(dist*mt.sqrt(1-(r[2]/dist)**2)),r[0],dist,deg(lat))
    try:
        lon = mt.acos(r[0]/(dist*mt.sqrt(1-(r[2]/dist)**2))) #mt.cos(lat)))
    except ValueError:
        print('Correcting numerical error.. -> we are at the edges of the math domain: {} is not in [-1,1]'.format(r[0]/(dist*mt.sqrt(1-(r[2]/dist)**2))))
        if r[0] > 0:
            lon = np.pi
        else:
            lon = -np.pi

    if r[1] < 0:
        lon = 2*np.pi-lon

    if output_in_deg:
        lat = deg(lat)
        lon = deg(lon)

    return np.array([lat,lon,h])


def LOS_2D(alt_tg,alts,T,P,gas_ok,ext_coef,Rpl=2575.0):
    """
    Calculates LOS path in the atm for given tangent altitude. Homogeneous atmosphere. No refraction.
    :param alt_tg: tangent altitude of the LOS (km)
    :param alts: altitude grid (km)
    :param T: temp. profile (K)
    :param P: pres. profile (hPa)
    :param ext_coef: Aerosol ext. coeff. in cm-1
    :param gas_ok: VMR profiles of selected gases (matrix n_gases x n_alts)
    :param Rpl: planetary radius
    :return:
    """

    n = P/(kb*T) # num. density in cm-3

    Rtoa = np.max(alts)
    step = 10.0 # step in km
    _LOS = np.array([1,0])
    R_0 = np.array([-(mt.sqrt((Rtoa+Rpl)**2-(Rpl+alt_tg)**2)),Rpl+alt_tg]) # first LOS point

    R = R_0 + step * _LOS #first step
    #print(R_0)
    #print(R)
    #print(LA.norm(R)-Rpl)
    z_los = np.array([Rtoa,LA.norm(R)-Rpl])
    R_los = np.vstack([R_0,R])
    steps = np.array([step])

    while LA.norm(R) < Rtoa+Rpl:
        R = R + step * _LOS
        R_los = np.vstack([R_los,R])
        z_los = np.append(z_los,LA.norm(R)-Rpl)
        #print(R)
        #print(LA.norm(R)-Rpl)
        steps = np.append(steps,step)

    Tau_aer = np.interp(z_los[:-1], alts, ext_coef)
    Tau_aer = Tau_aer * steps * 1e5
    temps = np.interp(z_los[:-1], alts, T)
    nlos = np.interp(z_los[:-1], alts, n)
    press = np.interp(z_los[:-1], alts, np.log(P))
    press = np.exp(press)
    gases = np.zeros(len(z_los[:-1]))
    Rcols = np.zeros(len(z_los[:-1]))
    for gas in gas_ok:
        VMRs = np.interp(z_los[:-1], alts, gas)
        Rcol = VMRs * nlos * steps * 1e5
        gases = np.vstack([gases,VMRs])
        Rcols = np.vstack([Rcols,Rcol])
    gases = gases[1:,:]
    Rcols = Rcols[1:,:]

    return z_los[:-1],steps,temps,press,gases,Rcols,Tau_aer


def hydro_P(z,T,MM,P_0=None,R=Rtit,M=Mtit):
    """
    Calculates hydrostatic pressure in hPa, given temp. profile.
    :param z: altitude (km)
    :param T: temperature (K)
    :param MM: mean molecular mass (amu)
    :param P_0: Surface pressure (hPa)
    :param R: Planetary radius (km)
    :param M: Planetary mass (kg)
    :return: P, pressure profile on z grid
    """
    P_huy = 1.4612e3 #hPa

    if P_0 is None:  # define P_0 as the huygens pressure
        P_0 = P_huy
    reverse = False
    if z[1] < z[0]: # reverse order if z decreasing
        reverse = True
        z = z[::-1]
        T = T[::-1]
        MM = MM[::-1]
    if np.size(MM) == 1: # if MM is a scalar, then multiply for a np.ones vector
        mu = MM
        MM = mu*np.ones(len(z))

    R = R*1e3 # from km to m
    z = z*1e3
    MM = MM*1e-3 # from amu to kg/mol

    g = c_G*M/(R+z)**2
    print(g[0])
    #g=g*1.352/g[0]

    HH = MM*g/(c_R*T)

    P = np.zeros(len(z))
    P2 = P
    P[0] = P_0
    for i in range(1,len(z)):
        # dz = z[i]-z[i-1]
        # int = HH[i-1]*dz+0.5*dz*(HH[i]-HH[i-1])  # integrazione stupida
        # P[i] = P[i-1]*np.exp(-int)
        int = np.trapz(HH[0:i+1],x=z[0:i+1])
        P2[i] = P[0]*np.exp(-int)

    if reverse:
        P2 = P2[::-1]

    return P2


def findT(alt,alt_atm,temp,diff=1.0):
    """
    Finds T at altitude alt.
    :return:
    """
    T = 0
    for tt,altu in zip(temp,alt_atm):
        if(abs(alt-altu)<diff):
            T = tt

    return T


def find_near(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def find_incl(array,value1,value2):
    ok = np.all([value2 >= array, array >= value1], axis=0)
    return ok

#################################################################################
##                                                                            ###
##                                                                            ###
##       OTHER FUNCTIONS                                                     ####
##                                                                            ###
##                                                                            ###
#################################################################################


def BB(T,w):
    """
    Black body at temp. T in units of nW/(cm2*cm-1).
    :param T: Temperature
    :param w: Wavenumber (cm-1)
    :return:
    """
    rc1 = 1.1904e-3
    rhck = 1.4388

    BB = rc1 * w**3 / (mt.exp(w*rhck/T)-1)

    return BB


def BB_nm(T,w):
    """
    Black body at temp. T in units of W/(m2*nm).
    :param T: Temperature
    :param w: Wavelength (nm)
    :return:
    """
    rc1 = 1.1904
    rhck = 1.4388

    if T*w > 5e5 :
        BB = rc1 * mt.pow((1.e4/w),5) / (mt.exp(1.e7/w*rhck/T)-1)
    else:
        BB = rc1 * mt.pow((1.e4/w),5) * mt.exp(-1.e7/w*rhck/T)

    return BB


def pvalue(i,n):
    from scipy.stats import binom
    p = 2*binom.cdf(i,n,0.5)
    return p
