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
import time
import spect_main_module as smm
import spect_classes as spcl
import operator
import psutil

import Tkinter
import tkMessageBox

#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2
kbc = const.k/(const.h*100*const.c) # 0.69503

def check_free_space(cart, threshold = 0.05):
    fract = (1.0*os.statvfs(cart).f_bfree)/os.statvfs(cart).f_blocks
    print('Still {}% of disk space free'.format(int(fract*100.)))
    if fract < threshold:
        tkMessageBox.showwarning(title = 'aaaaaaaaaaaaahhhhhh', message = "Available disk space is below {}%. Free some space or kill process.".format(int(threshold*100)))
        #top = Tkinter.Tk(title = 'ATTENTION!!!', message = "Available disk space is below {}%. Free some space or kill process.".format(int(threshold*100)))
        #B1 = Tkinter.Button(top, text = 'Go!')
        #B1.pack()
        #top.mainloop()
    return fract


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

    def distance(self, other, units = 'cm'):
        """
        Points are stored in km. units is the units of the output, usually cm to be used in radtran.
        """
        su = 0.0
        for un,du in zip(self.Cartesian(), other.Cartesian()):
            su += (un-du)**2
        if units == 'cm':
            return mt.sqrt(su)*1.e5
        elif units == 'km':
            return mt.sqrt(su)
        else:
            raise ValueError('Unknown unit')


class LineOfSight(object):
    """
    Class to represent the geometry of an observation.
    """

    def __init__(self, spacecraft_coords, second_point, delta_ang = None, rot_plane_ang = None):
        """
        The coordinates of first and second point (Coords objects).
        : delta_ang : (optional). The los vector is rotated by delta_ang (in rad) in the plane containing the original vector and the planet center. If rot_plane_ang is set, the rotation of the los vector is performed in a plane rotated counterclockwise by rot_plane_ang.
        """
        self.starting_point = copy.deepcopy(spacecraft_coords)
        self.second_point = copy.deepcopy(second_point)
        self.atm_quantities = dict()
        self.delta_ang = delta_ang
        self.rot_plane_ang = rot_plane_ang

        return

    def calc_LOS_vector(self):#, planet, ellipsoidal = False):
        """
        Adds to the LOS object a parameter vector that points from the Spacecraft to the observed point on the planet.
        """
        p1 = self.starting_point
        p2 = self.second_point
        if self.delta_ang is not None:
            dist = p1.distance(p2, units = 'km')
            delta_h = mt.tan(self.delta_ang)*dist
            olc2 = p2.Spherical()
            olc2[2] += delta_h
            p2 = Coords(olc2, s_ref='Spherical')
            if self.rot_plane_ang is not None:
                raise ValueError('sto cazzone non ha ancora scritto niente qua')

        c1 = copy.deepcopy(p1.Cartesian())
        c2 = copy.deepcopy(p2.Cartesian())

        _LOS = (c2-c1)/LA.norm(c2-c1) # Unit LOS vector
        # print(c2-c1,LA.norm(c2-c1))
        # print(p1.Cartesian(), self.second_point.Cartesian(), p2.Cartesian())

        self._LOS = _LOS

        return _LOS


    def details(self):
        print('Observer point: {}, {}'.format(self.starting_point.Cartesian(), self.starting_point.Spherical()))
        print('Second point: {}, {}'.format(self.second_point.Cartesian(), self.second_point.Spherical()))
        print('delta_ang: {}, rot_plane_ang: {}'.format(self.delta_ang, self.rot_plane_ang))
        try:
            print('LOS vector: {}'.format(self._LOS))
        except:
            print('LOS vector: {}'.format(self.calc_LOS_vector()))

        return

    def calc_along_LOS(self, atmosphere, profname = None, curgod = False, set_attr = False, set_attr_name = None):
        """
        Returns a vector with prof values on the LOS.
        ###################### WIPPPPPP
        CURGOD TO BE INTRODUCED
        method available: simple, curgod
        """

        if profname is None:
            nomi = atmosphere.names
            if len(nomi) > 1:
                raise ValueError('specify the profile to be used, among these: {}'.format(nomi))
            else:
                profname = nomi[0]

        try:
            points = self.intersections
        except:
            points = self.calc_atm_intersections(planet)

        quant = []

        # for point1,point2 in zip(points[:-1],points[1:]):
        #     point = Coords((point1.Cartesian()+point2.Cartesian())/2)
        #     quant.append(atmosphere.calc(point.Spherical()[2],profname))
        #     if curgod:
        #         print('Not yet available! Doing simple interpolation..')
        for point in points:
            quant.append(atmosphere.calc(point.Spherical()[2],profname))

        #print('PUPPAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ---- questo va riscrittooooooooooooo osososooso sooooo: non è generalizzato per profilo a più dimensioni tipo lat lon alt')
        if set_attr:
            if set_attr_name is None:
                set_attr_name = profname
            self.atm_quantities[set_attr_name] = np.array(quant)
            if 'temp' in self.atm_quantities.keys() and 'pres' in self.atm_quantities.keys():
                self.atm_quantities['ndens'] = num_density(self.atm_quantities['pres'], self.atm_quantities['temp'])

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


    def calc_atm_intersections(self, planet, delta_x = 1.0, start_from_TOA = True, refraction = False, LOS_order = 'radtran', verbose = False):
        """
        Calculates the coordinates of the points that the LOS crosses in the atmosphere. If start_from_TOA is True (default), the LOS path starts at the first intersection with the atmosphere, instead it starts from the spacecraft.
        The default order is from the closest point to spacecraft to the other side (LOS_order = 'radtran'). Order can be set to 'photon'.

        NO REFRACTION INCLUDED!! work in progress.....
        """

        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            #print(cazzillo)

        TOA_R = planet.atm_extension + planet.radius
        R = planet.radius

        if start_from_TOA:
            #print('Qui')
            #print(self.starting_point.Cartesian())
            point_0, i_type = self.intersect_shell(planet, self.starting_point.Cartesian(), TOA_R)
            if i_type != 'Ingress':
                raise ValueError('No intersection with shell!')
            #print('Quo')
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
            point = self.move_along_LOS(planet, point, delta_x, refraction = refraction)
            inside = (LA.norm(point) < TOA_R)
            #print(inside, point)
            surface_hit = (LA.norm(point) < R)
            if surface_hit:
                point = self.move_along_LOS(planet, point, delta_x, refraction = refraction, backward = True)
                point, i_type = self.intersect_shell(planet, point, R)
                if i_type != 'Ingress':
                    raise ValueError('No intersection with shell!')
                los_points.append(Coords(point, R = planet.radius))
                if verbose: print('Hit the surface.. Stopping ray path.')
                break
            elif inside and not surface_hit:
                po = Coords(point, R = planet.radius)
                los_points.append(copy.deepcopy(po))
                if verbose: print('Still inside at {} (spherical)'.format(po.Spherical()))

        #l'integrazione va su tau, devo partire dalla "fine" della LOS, la parte più vicina a me.
        if LOS_order == 'photon':
            los_points = los_points[::-1]

        self.intersections = los_points

        return los_points


    def calc_radtran_steps(self, planet, lines, max_opt_depth = None, max_T_variation = 5.0, max_Plog_variation = 2.0, calc_derivatives = False, bayes_set = None, verbose = False):
        """
        Calculates the optimal step for radtran calculation.
        """

        try:
            gigi = self.intersections
            gigi = self.temp
            gigi = self.pres
            for gas in planet.gases:
                gigi = self.gas
        except:
            self.calc_atm_intersections(planet)
            self.calc_along_LOS(planet.atmosphere, profname = 'temp', set_attr = True)
            self.calc_along_LOS(planet.atmosphere, profname = 'pres', set_attr = True)

        tvibs = []
        for gas in planet.gases:
            conc_gas = self.calc_abundance(planet, gas, set_attr = True)

            gasso = planet.gases[gas]
            for iso in gasso.all_iso:
                isomol = getattr(gasso, iso)
                if not isomol.is_in_LTE:
                    for lev in isomol.levels:
                        levello = getattr(isomol, lev)
                        tvi = self.calc_along_LOS(levello.vibtemp)
                        tvibs.append(tvi)

        # Calcolo shape e value della linea più intensa a diverse temps.
        min_temp = np.min(self.atm_quantities['temp'])
        max_temp = np.max(self.atm_quantities['temp'])


        abs_max = dict()
        for mol_name, molec in zip(planet.gases.keys(), planet.gases.values()):
            lines_mol = [lin for lin in lines if lin.Mol == molec.mol]

            if len(lines_mol) == 0:
                abs_max[mol_name] = 0.0
                continue

            if len(lines_mol) > 50:
                essesss = [lin.Strength for lin in lines_mol]
                essort = np.sort(np.array(essesss))[-50]
                lines_sel = [lin for lin in lines_mol if lin.Strength >= essort]
            else:
                lines_sel = lines_mol

            essesss = [lin.CalcStrength_from_Einstein(max_temp)[0] for lin in lines_sel]
            # Estimating maximum height of the line peak
            lin = lines_sel[0]
            shape = lin.MakeShapeLine(min_temp, 1.e-8)
            peak_val = np.max(shape.spectrum)
            abs_max[mol_name] = peak_val*max(essesss)

        # adesso parto con gli step
        opt_depth_step = 0.0
        step = 0.0
        num_orig = 0
        point_prev = self.intersections[0]
        point_orig = self.intersections[0]

        self.radtran_steps = dict()
        self.radtran_steps['step'] = []
        self.radtran_steps['max_opt_depth'] = []
        self.radtran_steps['indices'] = []
        self.radtran_steps['temp'] = dict()
        self.radtran_steps['pres'] = dict()
        self.radtran_steps['ndens'] = dict()

        for gas in planet.gases:
            self.radtran_steps['ndens'][gas] = []
            self.radtran_steps['temp'][gas] = []
            self.radtran_steps['pres'][gas] = []

        if calc_derivatives:
            self.radtran_steps['deriv_factors'] = dict()
            for cos in bayes_set.sets.values():
                self.radtran_steps['deriv_factors'][cos.name] = dict()
                for par in cos.set:
                    self.radtran_steps['deriv_factors'][cos.name][par.key] = []

        end_LOS = False
        num = 0

        stp_tot = 0.0
        estim_tot_depth = 0.0

        cond_names = np.array(['temp','pres','tvib','opt_depth'])
        while not end_LOS:
            num += 1
            point_prev = self.intersections[num-1]
            point = self.intersections[num]

            step_add = point_prev.distance(point, units = 'cm')
            step += step_add
            abstop = max([abs_max[gas]*self.atm_quantities[gas][num] for gas in planet.gases])
            opt_depth_step += abstop*step_add

            #if verbose: print(num, step, opt_depth_step)

            t_var = max(self.atm_quantities['temp'][num_orig:num])-min(self.atm_quantities['temp'][num_orig:num])
            p_var_log = np.log(max(self.atm_quantities['pres'][num_orig:num])/min(self.atm_quantities['pres'][num_orig:num]))
            tvibs_var = []
            for tt in tvibs:
                tvibs_var.append(max(tt[num_orig:num])-min(tt[num_orig:num]))
            tvibs_var = np.array(tvibs_var)

            stepping = False

            cond = np.array([False, False, False, False])
            if t_var > max_T_variation: cond[0] = True
            if p_var_log > max_Plog_variation: cond[1] = True
            if np.any(tvibs_var > max_T_variation): cond[2] = True
            if max_opt_depth is not None:
                if opt_depth_step > max_opt_depth: cond[3] = True

            # Check T,P,Tvib variation
            # Check optical_depth
            if np.any(cond):
                num -= 1
                point = point_prev
                step -= step_add
                opt_depth_step -= abstop*step_add
                stepping = True

            if num == len(self.intersections)-1:
                stepping = True
                end_LOS = True

            if num == num_orig:
                raise ValueError('RadStep is thick with 1 step! raise the max_opt_depth threshold or lower the LOS step length..')

            if stepping:
                if verbose: print('stepping {:5d} <-> {:5d} of {:5d}. z0: {:8.3f} zf: {:8.3f}. Op_d: {:8.3f}. Trig: {}.'.format(num_orig, num, len(self.intersections), point_orig.Spherical()[2], point.Spherical()[2], opt_depth_step, cond_names[cond]))

                estim_tot_depth += opt_depth_step
                stp_tot += step

                self.radtran_steps['step'].append(step)
                self.radtran_steps['max_opt_depth'].append(opt_depth_step)
                self.radtran_steps['indices'].append([num_orig, num])
                for gas in planet.gases:
                    nd,ti = CurGod(self.atm_quantities[gas][num_orig:num+1],self.atm_quantities['temp'][num_orig:num+1])
                    nd,pi = CurGod(self.atm_quantities[gas][num_orig:num+1],self.atm_quantities['pres'][num_orig:num+1])
                    self.radtran_steps['ndens'][gas].append(nd)
                    self.radtran_steps['pres'][gas].append(pi)
                    self.radtran_steps['temp'][gas].append(ti)
                    # non-LTE part
                    gasso = planet.gases[gas]
                    for iso in gasso.all_iso:
                        isomol = getattr(gasso, iso)
                        if not isomol.is_in_LTE:
                            for lev in isomol.levels:
                                levello = getattr(isomol, lev)
                                tvi = self.calc_along_LOS(levello.vibtemp)
                                nd,tvi = CurGod(self.atm_quantities[gas][num_orig:num+1],tvi[num_orig:num+1])
                                try:
                                    levello.local_vibtemp.append(tvi)
                                except:
                                    levello.add_local_vibtemp(tvi)

                ndtot, _ = CurGod(self.atm_quantities['ndens'][num_orig:num+1], np.ones(num+1-num_orig))
                cdtot = ndtot*step

                if calc_derivatives:
                    for cos in bayes_set.sets.values():
                        deriv_set = self.radtran_steps['deriv_factors'][cos.name]
                        gas = cos.name
                        if verbose: print('gssss ', gas)
                        for par in cos.set:
                            masklos = self.calc_along_LOS(par.maskgrid)
                            nd, cg_mask = CurGod(self.atm_quantities[gas][num_orig:num+1], masklos[num_orig:num+1])
                            deriv_set[par.key].append(cdtot*cg_mask)
                            if verbose: print('dssss ', par.key, cg_mask)

                num_orig = num
                point_orig = self.intersections[num]
                step = 0.0
                opt_depth_step = 0.0

        print('Estimated total optical depth: {}'.format(estim_tot_depth))
        print('total length of LOS: {}'.format(stp_tot))

        return


    def calc_abundance(self, planet, gas, set_attr = False):
        """
        Calculates abundance (num density) of the gas among planet.gases along the LOS. vmr in absolute fraction!
        """
        vmr_gas = self.calc_along_LOS(planet.gases[gas].abundance)

        ndens = []
        #print('puppa')
        #print(self.atm_quantities)
        for P, T, vmr in zip(self.atm_quantities['pres'], self.atm_quantities['temp'], vmr_gas):
            nd = num_density(P, T, vmr)
            ndens.append(nd)

        if set_attr:
            self.atm_quantities[gas] = np.array(ndens)

        return np.array(ndens)


    def calc_optical_depth(self, wn_range, planet, lines, step = None, cartLUTs = None):
        """
        Calculates the optical depth along the LOS.
        """

        abs_opt_depth = smm.prepare_spe_grid(wn_range)
        emi_opt_depth = smm.prepare_spe_grid(wn_range)

        single_coeffs_abs = dict()
        single_coeffs_emi = dict()

        for gas in planet.gases:
            print(gas)
            gasso = planet.gases[gas]
            ndens = self.calc_abundance(planet, gas)
            #print('aaaaaaaaaaaaaaaaaargh ', ndens)
            iso_abs = []
            iso_emi = []
            #print(gasso.all_iso)
            for iso in gasso.all_iso:
                isomol = getattr(gasso, iso)
                print('Calculating mol {}, iso {}. Mol in LTE? {}'.format(isomol.mol,isomol.iso,isomol.is_in_LTE))
                if not isomol.is_in_LTE:
                    for lev in isomol.levels:
                        #print(lev)
                        levello = getattr(isomol, lev)
                        tvi = self.calc_along_LOS(levello.vibtemp)
                        levello.add_local_vibtemp(tvi)

                set_abs = []
                set_emi = []

                #print('Catulloneeeeeeee')
                abs_coeffs, emi_coeffs = smm.make_abscoeff_isomolec(wn_range, isomol, self.atm_quantities['temp'], self.atm_quantities['pres'], lines = lines, LTE = isomol.is_in_LTE, allLUTs = LUTS, store_in_memory = True)
                iso_ab = isomol.ratio
                for aboo, emoo, ndoo in zip(abs_coeffs,emi_coeffs, ndens):
                    #print(aboo,emoo,ndoo,iso_ab)
                    abs_coeff_tot = smm.prepare_spe_grid(wn_range)
                    emi_coeff_tot = smm.prepare_spe_grid(wn_range)
                    abs_coeff_tot.add_to_spectrum(aboo, Strength = iso_ab*ndoo)
                    emi_coeff_tot.add_to_spectrum(emoo, Strength = iso_ab*ndoo)
                    set_abs.append(abs_coeff_tot)
                    set_emi.append(emi_coeff_tot)

                iso_abs.append(set_abs)
                iso_emi.append(set_emi)
            single_coeffs_abs[gas] = copy.deepcopy(iso_abs)
            single_coeffs_emi[gas] = copy.deepcopy(iso_emi)

        steps = [step]*len(single_coeffs_abs)
        for gas in planet.gases:
            for isoco in single_coeffs_abs[gas]:
                self.spectralsum_along_LOS(abs_opt_depth, isoco, strengths = steps)
            for isoco in single_coeffs_emi[gas]:
                self.spectralsum_along_LOS(emi_opt_depth, isoco, strengths = steps)

        return abs_opt_depth, emi_opt_depth, single_coeffs_abs, single_coeffs_emi


    def radtran(self, wn_range, planet, lines, cartLUTs = None, calc_derivatives = False, bayes_set = None, initial_intensity = None, cartDROP = None, tagLOS = None, debugfile = None, useLUTs = False, LUTS = None, radtran_opt = None):
        """
        Calculates the radtran along the LOS. step in km.
        """

        if tagLOS is None:
            tagLOS = 'LOS'

        if cartDROP is None:
            cartDROP = 'stuff_'+smm.date_stamp()
            if not os.path.exists(cartDROP):
                os.mkdir(cartDROP)
            cartDROP += '/'

        try:
            gigi = self.radtran_steps['step']
            if calc_derivatives:
                gigi = self.radtran_steps['deriv_factors']
        except:
            if radtran_opt is None:
                self.calc_radtran_steps(planet, lines, calc_derivatives = calc_derivatives, bayes_set = bayes_set)
            else:
                self.calc_radtran_steps(planet, lines, calc_derivatives = calc_derivatives, bayes_set = bayes_set, **radtran_opt)

        spe_zero = smm.prepare_spe_grid(wn_range)

        if initial_intensity is None:
            intensity = spcl.SpectralIntensity(spe_zero.spectrum, spe_zero.spectral_grid)
        else:
            intensity = copy.deepcopy(initial_intensity)

        all_molecs_abs = dict()
        all_molecs_emi = dict()

        # CALCULATING SINGLE GAS-iso ABS and emi

        for gas in planet.gases:
            gasso = planet.gases[gas]

            temps = self.radtran_steps['temp'][gas]
            press = self.radtran_steps['pres'][gas]

            all_iso_abs = dict()
            all_iso_emi = dict()
            print(gasso.all_iso)
            for iso in gasso.all_iso:
                check_free_space(cartDROP)
                isomol = getattr(gasso, iso)
                if len([lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso]) == 0:
                    print('Skippin gas {} {}, no lines found'.format(gas, iso))
                    continue
                print('Calculating mol {}, iso {}. Mol in LTE? {}'.format(isomol.mol,isomol.iso,isomol.is_in_LTE))
                #print('Catulloneeeeeeee')
                abs_coeffs, emi_coeffs = smm.make_abscoeff_isomolec(wn_range, isomol, temps, press, lines = lines, LTE = isomol.is_in_LTE, allLUTs = LUTS, store_in_memory = True, cartDROP = cartDROP, tagLOS = tagLOS, useLUTs = useLUTs)

                all_iso_abs[iso] = copy.deepcopy(abs_coeffs)
                all_iso_emi[iso] = copy.deepcopy(emi_coeffs)

            all_molecs_abs[gas] = all_iso_abs
            all_molecs_emi[gas] = all_iso_emi

        # CALCULATING TOTAL ABS AND emi

        abs_coeff_tot = smm.AbsSetLOS(cartDROP+'abscoeff_tot_'+tagLOS+'.pic')
        emi_coeff_tot = smm.AbsSetLOS(cartDROP+'emicoeff_tot_'+tagLOS+'.pic')
        abs_coeff_tot.prepare_export()
        emi_coeff_tot.prepare_export()

        for num in range(len(self.radtran_steps['step'])):
            abs_coeff = smm.prepare_spe_grid(wn_range)
            emi_coeff = smm.prepare_spe_grid(wn_range)
            for gas in all_molecs_abs.keys():
                nd = self.radtran_steps['ndens'][gas][num]
                gasso = planet.gases[gas]
                for iso in all_molecs_abs[gas].keys():
                    check_free_space(cartDROP)
                    isomol = getattr(gasso, iso)
                    gigio_ab = all_molecs_abs[gas][iso]
                    gigio_em = all_molecs_emi[gas][iso]
                    if gigio_ab.temp_file is None:
                        gigio_ab.prepare_read()
                        gigio_em.prepare_read()
                    ab = gigio_ab.read_one()
                    em = gigio_em.read_one()
                    iso_ab = isomol.ratio

                    abs_coeff += ab * (nd*iso_ab)
                    emi_coeff += em * (nd*iso_ab)

            abs_coeff_tot.add_dump(abs_coeff)
            emi_coeff_tot.add_dump(emi_coeff)

        for gas in all_molecs_abs.keys():
            for iso in all_molecs_abs[gas].keys():
                all_molecs_abs[gas][iso].finalize_IO()
                all_molecs_emi[gas][iso].finalize_IO()

        abs_coeff_tot.finalize_IO()
        emi_coeff_tot.finalize_IO()

        steps = self.radtran_steps['step']
        single_intensities = dict()
        iso_intensities = dict()

        for gas in all_molecs_abs.keys():
            #print('catuuuuusppspsps: ', gas)
            all_iso_emi = all_molecs_emi[gas]
            all_iso_abs = all_molecs_abs[gas]
            ndens = np.array(self.radtran_steps['ndens'][gas])

            ret_set = None
            derivfa = None
            if calc_derivatives:
                if gas in bayes_set.sets.keys():
                    ret_set = bayes_set.sets[gas]
                    derivfa = self.radtran_steps['deriv_factors'][gas]
                    for par in ret_set.set:
                        par.add_hires_deriv(spcl.SpectralIntensity(spe_zero.spectrum, spe_zero.spectral_grid))

            for iso in all_molecs_abs[gas].keys():
                #print('catuuuuusppspsps: ', iso)
                emi_coeffs = all_iso_emi[iso]
                abs_coeffs = all_iso_abs[iso]
                isomol = getattr(planet.gases[gas], iso)
                iso_ab = isomol.ratio
                coso = self.radtran_single(intensity, abs_coeff_tot, abs_coeffs, emi_coeffs, steps, ndens, iso_ab = iso_ab, calc_derivatives = calc_derivatives, ret_set = ret_set, deriv_factors = derivfa, debugfile = debugfile)
                if calc_derivatives:
                    # ret_set_iso = coso[1]
                    # for par_iso, par in zip(ret_set_iso.set, ret_set.set):
                    #     par.hires_deriv += par_iso.hires_deriv
                    intens = coso[0]
                else:
                    intens = coso

                iso_intensities[iso] = copy.deepcopy(intens)
            single_intensities[gas] = copy.deepcopy(iso_intensities)

        for gas in all_molecs_abs.keys():
            for iso in all_molecs_abs[gas].keys():
                intensity += single_intensities[gas][iso]

        if calc_derivatives:
            return intensity, single_intensities, bayes_set
        else:
            return intensity, single_intensities


    def radtran_single(self, intensity, abs_coeff_tot, abs_coeff_gas, emi_coeff_gas, steps, ndens, iso_ab = 1.0, calc_derivatives = False, ret_set = None, deriv_factors = None, debugfile = None):
        """
        Integrates the formal solution of the radtran equation along the LOS.
        : abs_coeff_tot : the absorption coefficient of all gases in the atmosphere.
        : abs_coeff_gas : the abs coeff of the single gas/iso considered. Used only for derivatives.
        : emi_coeff_gas : the emi coeff of the single gas/iso considered. Used for radtran and derivatives.
        : steps : the set of steps of the LOS. In cm.
        : ndens : the set of partial column densities of the LOS. in cm-2.
        : iso_ab : the iso abundance. Default 1.0.
        : calc_derivatives : bool to turn on derivatives calculation
        : ret_set : the set of retrieval parameters connected with gas vmr.
        : deriv_factors : the derivatives of the gas number density with respect to the parameters (dictionary)
        """
        Ones = spcl.SpectralObject(np.ones(len(intensity.spectrum), dtype = float), intensity.spectral_grid)
        Zeros = spcl.SpectralObject(np.zeros(len(intensity.spectrum), dtype = float), intensity.spectral_grid)
        Gama_tot = copy.deepcopy(Ones)
        izero = copy.deepcopy(intensity)

        time0 = time.time()
        ii = 0
        # Summing step by step the contribution of the local Source function
        abs_coeff_tot.prepare_read()
        abs_coeff_gas.prepare_read()
        emi_coeff_gas.prepare_read()

        if calc_derivatives:
            Gama_strange = dict()
            for par in ret_set.set:
                Gama_strange[par.key] = copy.deepcopy(Zeros)

        print('TOTAL STEPS: {}'.format(abs_coeff_tot.counter))

        ii = 0
        for step, nd, num in zip(steps, ndens, range(len(steps))):
            print(step,nd)
            ii += 1
            ab_tot = abs_coeff_tot.read_one()
            em = emi_coeff_gas.read_one()
            ab = abs_coeff_gas.read_one()
            print('{} steps remaining'.format(abs_coeff_tot.remaining))

            max_depth = np.max(ab_tot.spectrum)*step
            #if(max_depth > 1.):
            #    print('Step {} is not optically thin. Max optical depth: {}'.format(ii, max_depth))
            time1 = time.time()

            tau = ab_tot*step
            gama = tau.exp_elementwise(exp_factor = -1.0)
            Source = (em*nd*iso_ab)/ab_tot
            Source.spectrum[np.isnan(Source.spectrum)] = 0.0
            Source.spectrum[np.isinf(Source.spectrum)] = 0.0
            #print('Questo è brutto! Cambia mettendo la source alla temp sua, vabbè è bruttino uguale eh..')

            unomenogama = gama*(-1.0)+1.0
            intensity += unomenogama*Source*Gama_tot

            if calc_derivatives:
                degamadeq = ab*gama*(-iso_ab)
                # questo per il pezzo non-LTE che non serve a nulla in realtà nella maggior parte dei casi
                cos = ((ab*(nd*iso_ab))/ab_tot)*(-1.0) + 1.0
                cos.spectrum[np.isnan(cos.spectrum)] = 0.0
                cos.spectrum[np.isinf(cos.spectrum)] = 0.0
                if np.any(cos.spectrum < 0.0) or np.any(cos.spectrum > 1.0):
                    raise RuntimeWarning('Strange things happening in the non-LTE derivative of the spectrum')
                    cos.spectrum[cos.spectrum > 1.0] = 1.0
                    cos.spectrum[cos.spectrum < 0.0] = 0.0
                deSdeq = Source*cos*degamadeq/nd

                for par in ret_set.set:
                    pezzo1 = Source*Gama_tot*degamadeq*(-deriv_factors[par.key][num])
                    pezzo2 = Source*unomenogama*Gama_strange[par.key]
                    pezzo3 = deSdeq*unomenogama*Gama_tot*deriv_factors[par.key][num]
                    par.hires_deriv += (pezzo1+pezzo2+pezzo3)
                    #print('ssssssssssssssssssssssss {} -> {}'.format(num, par.hires_deriv.max()))
                    Gama_strange[par.key] = Gama_strange[par.key]*gama + degamadeq*Gama_tot*deriv_factors[par.key][num]
                    if par is ret_set.set[-1]:
                        pl.figure(17)
                        pezzo1.plot(label = 'step {}'.format(num))
                        pl.figure(18)
                        pezzo2.plot(label = 'step {}'.format(num))
                        pl.figure(19)
                        pezzo3.plot(label = 'step {}'.format(num))
                        if iso_ab > 0.5 and debugfile is not None:
                            pickle.dump([num, pezzo1, pezzo2, pezzo3, Source, unomenogama, degamadeq, Gama_tot, Gama_strange], debugfile)

            Gama_tot *= gama

            # ok = (intensity.spectral_grid.grid > 2115.6286) & (intensity.spectral_grid.grid < 2115.6294)
            # print(('{:4d} '+9*'{:12.3e}' ).format(ii, step, nd, intensity.spectrum[ok][0],  ab.spectrum[ok][0], em.spectrum[ok][0], tau.spectrum[ok][0], tau_exp.spectrum[ok][0], Gama_tot.spectrum[ok][0], Source.spectrum[ok][0]))
            # pl.figure(41)
            # intensity.plot(label = 'Step {}'.format(ii))
            # pl.grid()
            # pl.figure(43)
            # Source.plot(label = 'Step {}'.format(ii))
            # pl.grid()
            # pl.figure(44)
            # tau.plot(label = 'Step {}'.format(ii))
            # pl.grid()
            print('Giro {} finito in {} sec'.format(ii, time.time()-time1))

        # summing the original intensity absorbed by the atmosphere
        print(type(izero), type(Gama_tot))
        intensity += izero*Gama_tot

        if calc_derivatives:
            for par in ret_set.set:
                par.hires_deriv += izero*Gama_strange[par.key]

        print('Finito radtran in {} s'.format(time.time()-time0))

        if calc_derivatives:
            return intensity, ret_set
        else:
            return intensity


    def spectralsum_along_LOS(self, abs_coeff, single_coeffs, strengths = None):
        if strengths is None:
            strengths = None*len(single_coeffs)
        #for step, coef, theforce in zip(self.intersections().steps, single_coeffs, strengths):
        for coef, theforce in zip(single_coeffs, strengths):
            abs_coeff.add_to_spectrum(coef, Strength = theforce)

        return abs_coeff


    def intersect_shell(self, planet, point, shell_radius, thres = 0.1, refraction = False):
        """
        Starting from point, finds the closer intersection with shell at radius shell_radius.
        DA AGGIUNGERE! : controllo di esistenza dell'intersezione
        # Ho 3 casi.
        # 1 - C'è almeno una intersezione nella forward direction, do come output la più vicina, che sia in ingress o egress.
        # 2 - Ci sono intersezioni solo in backward direction. Metto una flag backward e quando è falsa ritorno not found. Forse questa non è utile. Però vorrei che mi desse un output diverso da 3.
        # 3 - La LOS non interseca mai la shell. Not found.
        """

        # print('Starting from: {}'.format(point))
        try:
            _LOS = self._LOS
        except Exception as cazzillo:
            _LOS = self.calc_LOS_vector()
            #print(cazzillo)

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
            #print(cazzillo)

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


def num_density(P, T, vmr = 1.0):
    """
    Calculates num density. P in hPa, T in K, vmr in absolute fraction (not ppm!!)
    """
    n = vmr*P/(kb*T) # num. density in cm-3

    return n


def CurGod_ROZZO(ndens, quant, interp = 'lin', extension_factor = 10):
    """
    Curtis-Godson weighted average of atmospheric quantities. lin is linear interpolation, exp is exponential.
    """
    n = len(ndens)
    x = np.linspace(0.,1.,n)

    x_extended = np.linspace(0., 1.0, extension_factor*n)
    ndens_extended = np.interp(x_extended, x, np.log(ndens))
    ndens_extended = np.exp(ndens_extended)

    if interp == 'lin':
        quant_extended = np.interp(x_extended, x, quant)
    elif interp == 'exp':
        quant_extended = np.interp(x_extended, x, np.log(quant))
        quant_extended = np.exp(quant_extended)
    else:
        raise ValueError('interp type not valid')

    CG_nd = np.sum(ndens_extended)
    CG_quant = np.sum(quant_extended*ndens_extended)/CG_nd
    CG_nd = CG_nd/len(ndens_extended)

    return CG_nd, CG_quant


def CurGod(ndens, quant, interp = 'lin', x_grid = None):
    """
    Curtis-Godson weighted average of atmospheric quantities. lin is linear interpolation, exp is exponential.

    ------>>> if x_grid is None, REGULAR STEPS are assumed! <<<<----
    """
    from scipy import integrate

    #print('nddd', np.max(ndens), np.min(ndens), ndens)
    #print('qqqqq', np.max(quant), np.min(quant), quant)

    if x_grid is None:
        x_gri = np.linspace(0.,1.,len(ndens))
    else:
        x_gri = (x_grid-x_grid[0])/(x_grid[-1]-x_grid[0])

    def funz_ndens(x, ndens = ndens, x_gri = x_gri):
        ndi = np.interp(x, x_gri, np.log(ndens))
        ndi = np.exp(ndi)
        #print('funz_ndens')
        #print('i1 in', x, ndens, x_gri)
        #print('i1 out', ndi)
        return ndi

    def funz_quant_ndens(x, ndens = ndens, quant = quant, x_gri = x_gri, interp = interp):
        ndi = funz_ndens(x)
        if interp == 'lin':
            qui = np.interp(x, x_gri, quant)
        elif interp == 'exp':
            qui = np.interp(x, x_gri, np.log(quant))
            qui = np.exp(qui)
        else:
            raise ValueError('No interp method {}'.format(interp))

        #print('funz_quant_ndens')
        #print('i2 in', x, ndens, quant, x_gri)
        #print('i2 out', ndi)
        #print('i2 out', qui)
        return ndi*qui

    # try:
    CG_nd = integrate.quad(funz_ndens, 0., 1.)[0]
    CG_quant = integrate.quad(funz_quant_ndens, 0., 1.)[0]/CG_nd
    # except Exception as cazzillo:
        # print('oooooo')
        # xi = np.linspace(0.,1.,100000)
        # pl.ion()
        # pl.figure(27)
        # pl.title('ndens')
        # #pl.yscale('log')
        # pl.plot(x_gri, ndens)
        # pl.plot(xi, funz_ndens(xi))
        # pl.grid()
        # pl.figure(28)
        # pl.title('quant')
        # #if interp == 'exp':
        # #    pl.yscale('log')
        # pl.plot(x_gri, quant)
        # pl.grid()
        # pl.figure(29)
        # pl.title('ndi*quant')
        # #if interp == 'exp':
        # #    pl.yscale('log')
        # fu = funz_quant_ndens(xi)
        # print(np.any(np.isnan(fu)))
        # print(np.min(fu), np.max(fu))
        # pl.plot(xi, fu)
        # pl.grid()
        # print('vecchioo', CG_quant)
        # CGr = np.sum(fu)*(xi[1]-xi[0])/CG_nd
        # print('rozzooo', CGr)
        # mdens = np.mean(ndens)*np.ones(len(ndens))
        # CG_quant = integrate.quad(funz_quant_ndens, 0., 1., args = (mdens))[0]/CG_nd
        # print('nuovooo', CG_quant)
        # raise cazzillo

    return CG_nd, CG_quant


def funz_stup(x):
    return 4.0

def funz_parabola(x, a = 2.0, b = 1.0, c = 3.0):
    return a*x**2+b*x+c

def integ_parabola(x1, x2, a = 2.0, b = 1.0, c = 3.0):
    integ = (a*x2**3/3.0+b*x2**2/2.0+c*x2)-(a*x1**3/3.0+b*x1**2/2.0+c*x1)
    return integ

def funz_exp(x, a = 3.0, x0 = 2.0, H = 1.0):
    return a*np.exp(-(x-x0)/H)

def integ_exp(x1, x2, a = 3.0, x0 = 2.0, H = 1.0):
    integ = -H*(funz_exp(x2, a, x0, H)-funz_exp(x1, a, x0, H))
    return integ

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
        #print('Returning plane as function of (x,y) couples')
        def plane(x,y):
            z = point[2] - ( line[0]*(x-point[0]) + line[1]*(y-point[1]) ) / line[2]
            return np.array([x,y,z])
    else:
        #print('Returning plane as function of (x,z) couples')
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

        self.gases[gas.name] = copy.deepcopy(gas)
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
            setattr(self,key,copy.deepcopy(thing))
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

    def spacecraft(self):
        point = Coords([self.sub_obs_lat, self.sub_obs_lon, self.dist],s_ref='Spherical')
        print(point.Spherical())
        return point

    def limb_tg_point(self):
        point = Coords([self.limb_tg_lat, self.limb_tg_lon, self.limb_tg_alt],s_ref='Spherical')
        print(point.Spherical())
        return point

    def LOS(self, verbose = False, delta_ang = None, rot_plane_ang = None):
        spacecraft = self.spacecraft()
        second = self.limb_tg_point()

        linea1 = LineOfSight(spacecraft, second, delta_ang = delta_ang, rot_plane_ang = rot_plane_ang)
        if verbose:
            linea1.details()

        return linea1


class VIMSPixel(Pixel):
    """
    Some features specific to VIMS observations (FOV, wl windows)
    """
    def __init__(self, *args, **kwargs):
        Pixel.__init__(self, *args, **kwargs)
        # VIMS stuff
        self.FOV_angle_up = 0.5*1.e-3 # rad
        self.FOV_angle_down = -0.5*1.e-3 # rad
        return

    def low_LOS(self, verbose = False):
        return self.LOS(delta_ang = self.FOV_angle_down, verbose = verbose)

    def up_LOS(self, verbose = False):
        return self.LOS(delta_ang = self.FOV_angle_up, verbose = verbose)


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

    def __init__(self, mol, name=None):
        self.mol = mol   # mol number
        self.name = name # string representing the molecular name
        if name is None:
            self.name = find_molec_metadata(mol, 1)['mol_name']
        self.abundance = None
        self.n_iso = 0
        self.all_iso = []
        return

    def add_all_iso_from_HITRAN(self, lines = None, add_levels = True, custom_iso_ratios = None, n_max = None):
        """
        Creates and adds to Molec all isotopologues in HITRAN. The ones already added manually remain there. custom_iso_ratios is intended from the first isotopologue to be added.
        """
        if add_levels and lines is None:
            raise ValueError('Cannot load levels without a lines set. Please provide one')

        iso_names, iso_MM, iso_ratios = find_all_iso_HITRAN(self.mol)

        cos = len(self.all_iso)
        if n_max is None:
            iso_names = iso_names[cos:]
            iso_MM = iso_MM[cos:]
            iso_ratios = iso_ratios[cos:]
        else:
            iso_names = iso_names[cos:n_max]
            iso_MM = iso_MM[cos:n_max]
            iso_ratios = iso_ratios[cos:n_max]

        if custom_iso_ratios is not None:
            ok = len(custom_iso_ratios)
            iso_ratios = custom_iso_ratios + iso_ratios[ok:]

        num = cos + 1
        for ratio in iso_ratios:
            self.add_iso(num, ratio = ratio)
            num += 1

        if add_levels:
            for iso in self.all_iso[cos:]:
                isomol = getattr(self, iso)
                isomol.add_levels_from_HITRAN(lines)

        return

    def add_clim(self, profile):
        """
        Defines a climatology for the molecule. profile is the molecular abundance, has to be an AtmProfile object. type can be either 'VMR' or 'num_density (cm-3)'
        """
        if type(profile) is not AtmProfile:
            raise ValueError('Profile is not an AtmProfile object. Create it via Prof_ok = AtmProfile(profname, profile, grid, interp).')
        self.abundance = copy.deepcopy(profile)
        print('Added climatology for molecule {}'.format(self.name))

        return

    def add_iso(self, num, ratio = None, LTE = True):
        """
        Adds an isotope in form of an IsoMolec object.
        """
        string = 'iso_{:1d}'.format(num)

        try:
            iso = IsoMolec(self.mol, num, ratio=ratio, LTE=LTE)
        except:
            print('Molecule {} has no iso {} in HITRAN'.format(self.mol, num))
            return False
        setattr(self, string, iso)
        print('Added isotopologue {} for molecule {}'.format(string,self.name))
        self.n_iso += 1
        self.all_iso.append(string)

        return True

    def link_to_atmos(self, atmosphere, copy = False):
        """
        Creates a link to an AtmProfile object containing temp, pres, ...
        """
        if not copy:
            self.atmosphere = atmosphere
        else:
            self.atmosphere = copy.deepcopy(atmosphere)

        return


def find_all_iso_HITRAN(mol, metadatafile = './molparam.txt'):
    """
    Finds all isotopologues of molecule in the HITRAN list.
    """
    iso_names = []
    iso_MM = []
    iso_ratios = []
    for num in range(1,12):
        try:
            resu = find_molec_metadata(mol, num)
            print(resu)
            iso_names.append(resu['iso_name'])
            iso_MM.append(resu['iso_MM'])
            iso_ratios.append(resu['iso_ratio'])
        except:
            break

    return iso_names, iso_MM, iso_ratios


def find_levels_isomol_HITRAN(lines, mol, iso):
    """
    Finds all vibrational levels of molecule-iso in the HITRAN line list. Puts degenerate levels with different simmetries in the same level (adding simmetry to simmetries). Also finds the lower energy of the level, corresponding to the rotational fundamental.
    """
    lev_strings = []
    energies = []
    simmetries = []

    lines = [lin for lin in lines if lin.Mol == mol and lin.Iso == iso]

    levs = [lin.Lo_lev_str for lin in lines]+[lin.Up_lev_str for lin in lines]
    #print(levs)
    levs = np.unique(np.array(levs))
    #print(levs)

    minlevs = []
    for lev in levs:
        #print(lev)
        minst, vq, ot = extract_quanta_HITRAN(mol, iso, lev)
        if minst == '':
            continue
        minlevs.append(minst)

    #print(minlevs)
    minlevs = np.unique(np.array(minlevs))
    #print(minlevs)

    for lev in minlevs:
        lines_lev = [lin for lin in lines if minimal_level_string(mol, iso, lev) == lin.minimal_level_string_lo()]
        if len(lines_lev) > 0:
            energy = np.min(np.array([lin.E_lower for lin in lines_lev]))
            simmetries_lev = np.unique(np.array([lin.Lo_lev_str for lin in lines_lev]))
            simmetries_lev = list(simmetries_lev)
        else:
            lines_lev = [lin for lin in lines if minimal_level_string(mol, iso, lev) == lin.minimal_level_string_up()]
            energy = np.min(np.array([(lin.E_lower+lin.Freq) for lin in lines_lev]))
            simmetries_lev = np.unique(np.array([lin.Up_lev_str for lin in lines_lev]))
            simmetries_lev = list(simmetries_lev)
        if lev == '':
            continue
        iii = 0
        for sim in simmetries_lev:
            if sim.strip() == '':
                simmetries_lev.pop(iii)
            iii+=1
        lev_strings.append(lev)
        energies.append(energy)
        simmetries.append(simmetries_lev)

    print(lev_strings)
    print(energies)
    print(simmetries)

    energies = np.array(energies)
    lev_strings = np.array(lev_strings)
    simmetries = np.array(simmetries)

    order = np.argsort(energies)
    energies = list(energies[order])
    lev_strings = list(lev_strings[order])
    simmetries = list(simmetries[order])

    return lev_strings, energies, simmetries


def find_molec_metadata(mol, iso, filename = './molparam.txt'):
    """
    Loads molecular metadata from the molparam.txt HITRAN file. Returns a dict with: mol. name, iso. name, iso. ratio, iso. MM
    """

    if mol > 47:
        raise ValueError('There are only 47 molecs here.')

    #print('Looking for mol {}, iso {}'.format(mol,iso))
    resu = dict()

    infile = open(filename,'r')
    for i in range(mol):
        find_spip(infile)
        resu['mol_name'] = infile.readline().split()[0]
    for i in range(iso-1):
        linea = infile.readline()

    linea_ok = infile.readline().split()
    if linea_ok[0] == '#':
        raise ValueError('Iso not found.. you sure?')

    #print(linea_ok)
    resu['iso_name'] = linea_ok[0]
    resu['iso_ratio'] = float(linea_ok[1])
    resu['iso_MM'] = float(linea_ok[4])

    return resu


class IsoMolec(object):
    """
    Class to represent isotopes, with iso-ratios, vibrational levels, vib. temperatures, (for now).
    """

    def __init__(self, mol, iso, LTE = True, ratio=None):
        self.mol = mol   # mol number
        self.iso = iso   # iso number
        resu = find_molec_metadata(mol, iso)
        self.mol_name = resu['mol_name']
        self.name = resu['iso_name']
        self.MM = resu['iso_MM']   # mol mass
        if ratio is None:
            self.ratio = resu['iso_ratio']
        else:
            self.ratio = ratio   # isotopic ratio
        self.n_lev = 0 # number of levels specified
        self.levels = []
        self.is_in_LTE = LTE
        return

    def add_levels_from_HITRAN(self, lines):
        """
        Checks in HITRAN database all lines belonging to the isotopologue and finds all vibrational levels and their energies. Experimental, hope it works for every molecule.

        --------------- ATTENTION!!!! --------------------------------
        If the line list is incomplete (narrow wn_range) the level list will also be incomplete or have wrong informations (energies, simmetries...). Give here a LARGE set of lines, even all those in the database.
        """
        levels, energies, simmetries = find_levels_isomol_HITRAN(lines, self.mol, self.iso)
        self.add_levels(levels, energies, simmetries = simmetries)

        return levels, energies, simmetries

    def add_simmetries_levels(self, lines):
        levels, energies, simmetries = find_levels_isomol_HITRAN(lines, self.mol, self.iso)
        for lev in self.levels:
            levvo = getattr(self, lev)
            for levu, simm in zip(levels, simmetries):
                print(levu,levvo.minimal_level_string())
                if levvo.minimal_level_string() == minimal_level_string(self.mol, self.iso, levu):
                    levvo.add_simmetries(simm)
        return


    def add_levels(self, lev_strings, energies, vibtemps = None, degeneracies = None, simmetries = None, add_fundamental = False, T_kin = None):
        """
        Adds vibrational levels to selected isotopologue.
        """

        if degeneracies is None:
            degeneracies = len(lev_strings)*[-1]
        if simmetries is None:
            simmetries = len(lev_strings)*[[]]
        if vibtemps is None:
            vibtemps = len(lev_strings)*[None]
        else:
            for vit in vibtemps:
                print(type(vit))

        print(lev_strings)

        if add_fundamental:
            minstr = extract_quanta_HITRAN(self.mol, self.iso, lev_strings[0])[0]
            lev_0 = ''
            for lett in minstr:
                try:
                    num = int(lett)
                    lev_0 += '0'
                except:
                    lev_0 += lett

            energies.insert(0,0.0)
            lev_strings.insert(0, lev_0)
            simmetries.insert(0, None)
            degeneracies.insert(0, None)
            if vibtemps[0] is not None:
                vibtemps.insert(0, T_kin)
            else:
                vibtemps.insert(0, None)

        print(lev_strings)

        for levstr,ene,deg,sim,i,vib in zip(lev_strings,energies,degeneracies,simmetries,range(len(lev_strings)),vibtemps):
            print('Level <{}>, energy {} cm-1'.format(levstr,ene))
            stringaa = 'lev_{:02d}'.format(i)
            self.levels.append(stringaa)
            lev = Level(self.mol, self.iso, levstr, ene, degeneracy = deg, simmetry = sim)
            print('cerca ok qui sotto')
            if vib is not None:
                print('oooook')
                lev.add_vibtemp(vib)
            print(stringaa)
            setattr(self,stringaa,copy.deepcopy(lev))
            print(type(getattr(self,stringaa)))

            self.n_lev += 1

        return

    def add_level(self, lev_string, energy, degeneracy = -1, simmetry = [''], vibtemp = None):
        """
        Adds a single vib. level.
        """
        print('Level <{}>, energy {} cm-1'.format(lev_string,energy))
        string = 'lev_{:02d}'.format(self.n_lev)
        lev = Level(self.mol, self.iso, lev_string, energy, degeneracy = degeneracy, simmetry = simmetry)
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
            qlev = level.get_quanta()
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

    def __init__(self, mol, iso, levstring, energy, degeneracy = -1, simmetry = []):
        self.mol = mol
        self.iso = iso
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
        self.vibtemp = copy.deepcopy(profile)
        print('Added vibrational temperature to level <{}>'.format(self.lev_string))

        return

    def add_simmetries(self, simmetries):
        self.simmetry = simmetries
        return

    def add_local_vibtemp(self,vibtemp):
        """
        Adds attribute local_vibtemp to level. vibtemp could be either a scalar or a list/array.
        """
        try:
            len(vibtemp)
        except:
            vibtemp = [vibtemp]

        self.local_vibtemp = copy.deepcopy(vibtemp)
        return

    def add_Gcoeffs(self,Gcoeffs):
        """
        Adds attribute Gcoeffs to level. Gcoeffs is a dictionary with three values: "absorption", "ind_emission", "sp_emission". Each value is a SpectralGcoeff object corresponding to level.
        """
        if type(Gcoeffs) is dict:
            Gcoeffs = [Gcoeffs]
        self.Gcoeffs = copy.deepcopy(Gcoeffs)
        return

    def get_quanta(self):
        """
        Reads vibrational quanta and other quantum numbers. vib_quanta is a list, others is a dict.
        """

        qu, vib_quanta, others = extract_quanta_HITRAN(self.mol, self.iso, self.simmetry[0])

        if vib_quanta is None:
            raise ValueError('Not possible to extract vib_quanta for molecule {}, iso {}'.format(self.mol, self.iso))

        return vib_quanta, others

    def equiv(self, string_lev, check_minor_numbers = False):
        """
        Returns true if string_lev has the same quanta of Level. If check_simmetry is set to True, returns True only if also the simmetry is the same.
        """
        minst, vibq, oth = extract_quanta_HITRAN(self.mol, self.iso, self.simmetry[0])
        minst2, vibq2, oth2 = extract_quanta_HITRAN(self.mol, self.iso, string_lev)

        equiv = False
        if minst == minst2:
            equiv = True

        if check_minor_numbers:
            if oth != oth2:
                equiv = False

        return equiv

    def minimal_level_string(self):
        return minimal_level_string(self.mol, self.iso, self.lev_string)


def minimal_level_string(mol, iso, lev_string):
    if len(lev_string) == 15:
        minimal_level_string, qu, qi = extract_quanta_HITRAN(mol, iso, lev_string)
    else:
        minimal_level_string = lev_string.strip()

    return minimal_level_string


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


class AtmGrid(object):
    """
    The atmospheric grid. It is useful to separate it from AtmProfile, so one can have the control of the grid without getting lost in the np.meshgrid.
    : coord_names : list of names of the different dimensions.
    : coord_points_list : list of coordinate point lists, one for each dimension
    : coord_units : optional, list of units
    : box_names : optional, dictionary: {coord_name, box_names}. Used to name grid points in a different way. Useful for box grids, for example latitudes ('SP', 'MLS', 'EQ', ..).
    """
    def __init__(self, coord_names, coord_points_list, coord_units = None, hierarchy = None, box_names = None):
        if type(coord_names) is not list:
            coord_names = [coord_names]
            coord_points_list = [coord_points_list]
            if coord_units is not None:
                coord_units = [coord_units]
        self.ndim = len(coord_names)
        self.coords = dict()
        self.internal_order = dict()
        self.units = dict()

        ii = 0
        for nam, coord in zip(coord_names, coord_points_list):
            self.coords[nam] = copy.deepcopy(np.array(coord))
            self.internal_order[nam] = ii
            ii += 1

        if coord_units is None:
            coord_units = [None]*len(coord_names)
        for nam, unit in zip(coord_names, coord_units):
            self.units[nam] = unit

        if len(coord_names) == 1:
            self.grid = np.array(coord_points_list)
        else:
            cos = np.meshgrid(*coord_points_list)
            cosut = []
            for coui in cos:
                cosut.append(np.array(coui.T))
            self.grid = cosut

        if hierarchy is None:
            hierarchy = np.arange(self.ndim)

        self.hierarchy = np.array(hierarchy)
        self.box_names = copy.deepcopy(box_names)

        return

    def order(self):
        sorted_names = sorted(self.internal_order.items(), key=operator.itemgetter(1))
        for cos in sorted_names:
            print('{}: {}'.format(cos[1],cos[0]))
        return sorted_names

    def names(self):
        sorted_names = sorted(self.internal_order.items(), key=operator.itemgetter(1))
        names = [cos[0] for cos in sorted_names]
        return names

    def range(self):
        ranges = []
        sorted_names = sorted(self.internal_order.items(), key=operator.itemgetter(1))
        for [nam, num] in sorted_names:
            coo = self.coords[nam]
            mino = min(coo)
            maxo = max(coo)
            steps = []
            for co1,co2 in zip(coo[:-1],coo[1:]):
                steps.append(co2-co1)
            steps = np.array(steps)
            steps = np.unique(steps)
            if len(steps) > 1:
                stepo = 'irregular'
            else:
                stepo = steps[0]
            rangio = [mino,maxo]
            ranges.append((nam, rangio))
            print('{}: {} -> {}  --  step: {}'.format(num,nam,rangio,stepo))
        return ranges

    def merge(self, other):
        """
        Merges two grids with different dimensions returning an AtmGrid with all dimensions of both grids. Order is self first.
        """
        names = self.names()+other.names()

        coords = [self.coords[nam] for nam in self.names()]+[other.coords[nam] for nam in other.names()]

        gigi = AtmGrid(names, coords)

        return gigi

    def points(self):
        """
        Gives a list of all points in grid. Each point is a list of the coordinate values in the right order. The order of the points in the list is that of np.flatten() function on a slice of self.grid.
        """

        for cos in self.grid:
            if 'view' in locals():
                for el,vi in zip(cos.flatten(), view):
                    vi.append(el)
            else:
                view = []
                for el in cos.flatten():
                    view.append([el])

        return view


class AtmProfile(object):
    """
    Class to represent atmospheric profiles. Contains an AtmGrid object and a routine to get the interpolated value of the profile in the grid.
    Contains a set of interp strings to determine the type of interpolation.
    Input:
        grid -----------> AtmGrid object
        profile --------> array or list
        profname ----------> name of the profile (es. 'temp', 'pres', ..)
        interp ---------> list of strings : how to interpolate in given dimension? Accepted values: 'lin' - linear interp, 'exp' - exponential interp o 'box' - nearest neighbour
    """

    def __init__(self, grid, profile, profname, interp, descr=None):
        profile = np.array(profile)
        if type(grid) is not AtmGrid:
            raise ValueError('grid is not of type AtmGrid. Make grid first.. -> sbm.AtmGrid(coord_names, coord_points)')
        grid = copy.deepcopy(grid)
        self.descr = descr
        setattr(self, profname, copy.deepcopy(profile))
        self.ndim = grid.ndim

        self.names = []
        self.grid = copy.deepcopy(grid)
        self.interp = dict()

        if self.ndim != self.grid.ndim: raise ValueError('profile and grid have different dimensions!')

        self.add_profile(profile, profname, interp)

        return

    def get(self, profname):
        """
        Gets a single atmprofile object from an AtmProfile with more profiles.
        """
        nuprof = AtmProfile(self.grid, getattr(self, profname), profname, self.interp[profname])
        return nuprof

    def keys(self):
        return self.names

    def items(self):
        return zip(self.names, [getattr(self, nam) for nam in self.names])

    def add_profile(self, profile, profname, interp):
        """
        Adds a new profile to the atmosphere.
        """
        profile = np.array(profile)

        if self.ndim != profile.ndim: raise ValueError('New profile has dim {} instead of {}!'.format(profile.ndim, self.ndim))

        if np.shape(profile) != np.shape(self.grid.grid[0]):
            print('WARNING: profile shape is different! Trying to invert the order')
            print(np.shape(profile), self.grid.grid[0].shape)
            profile = profile.T
            print(np.shape(profile))
            if np.shape(profile) != np.shape(self.grid.grid[0]):
                raise ValueError('Profile has the wrong shape')
            else:
                print('Reshuffling successful, going on..')

        setattr(self, profname, copy.deepcopy(profile))

        self.names.append(profname)

        if type(interp) is str:
            interp = [interp]
        for ino in interp:
            nam = ['lin','exp','box']
            try:
                nam.index(ino)
            except ValueError:
                raise ValueError('{} is not a possible interpolation scheme. Possible schemes: {}'.format(ino,nam))

        interp = np.array(interp)
        self.interp[profname] = interp

        for val in np.unique(self.grid.hierarchy):
            oi = (self.grid.hierarchy == val)
            if len(np.unique(interp[oi])) > 1:
                raise ValueError('Can not interpolate 2D between int and exp coordinates!')
            elif np.unique(interp[oi])[0] == 'exp' and len(interp[oi]) > 1:
                raise ValueError('Can not interpolate 2D between exp coordinates!')
        return

    def __getitem__(self, key):
        """
        getitem works on coordinate values. for Example, if I have a grid in altitudes that goes from 1. to 100. and in lats from 30 to 80, and i want to extract alts from 20 to 30 and lats from 50 to 70, I have to write: new_prof = prof[20.:30.,50:70]. If I set the step as well, this tells the new interpolation step of the profile, based.
        """

        if len(key) != self.ndim:
            raise ValueError('number of dimension is {}, but {} were defined in getitem'.format(self.ndim, len(key)))

        for slic in key:
            if slic.step is not None:
                return self.__getslice__(key)

        cond_tot = np.ones(np.shape(self.grid.grid[0]),dtype = bool)
        for coord, slic in zip(self.grid.grid, key):
            #print(coord, slic)
            if slic.start is not None and slic.stop is not None:
                cond = (coord >= slic.start) & (coord <= slic.stop)
            elif slic.start is None and slic.stop is not None:
                cond = (coord <= slic.stop)
            elif slic.start is not None and slic.stop is None:
                cond = (coord >= slic.start)
            else:
                continue
            cond_tot = cond_tot & cond

        names = []
        coords = []
        for nam, coord in zip(self.grid.names(), self.grid.grid):
            names.append(nam)
            coo = np.unique(coord[cond_tot])
            coords.append(coo)

        nugrid_ok = AtmGrid(names, coords)

        for profname in self.names:
            prof = getattr(self, profname)
            proflin = prof[cond_tot]
            nuprof = proflin.reshape(np.shape(nugrid_ok.grid[0]))
            if 'profnew' in locals():
                profnew.add_profile(nuprof, profname, self.interp[profname])
            else:
                profnew = AtmProfile(nugrid_ok, nuprof, profname, self.interp[profname])

        return profnew


    def __getslice__(self, key):
        print('slice!')
        print('non lho scrittaaaaaaaaaaaa')
        print(key)
        return


    def __add__(self, other):
        """
        Sums two profiles. They have to contain the same number of profiles. If they contain more than one profile the names have to be the same.
        """
        profname1 = self.names[0]

        if isinstance(other, AtmProfile):
            if len(self.names) == 1:
                profname2 = other.names[0]
            else:
                profname2 = profname1

            if self.grid.grid.shape != other.grid.grid.shape:
                raise ValueError('Cannot sum two profiles with different grids.')

            prof = getattr(self, profname1)+getattr(other, profname2)
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)+getattr(other, nam)
                    profnew.add_profile(prof, nam, self.interp[nam])
        else:
            prof = getattr(self, profname1)+other
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)+other
                    profnew.add_profile(prof, nam, self.interp[nam])

        return profnew

    def __sub__(self, other):
        """
        Sums two profiles. They have to contain the same number of profiles. If they contain more than one profile the names have to be the same.
        """
        profname1 = self.names[0]

        if isinstance(other, AtmProfile):
            if len(self.names == 1):
                profname2 = other.names[0]
            else:
                profname2 = profname1

            if self.grid.grid.shape != other.grid.grid.shape:
                raise ValueError('Cannot sum two profiles with different grids.')

            prof = getattr(self, profname1)-getattr(other, profname2)
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)-getattr(other, nam)
                    profnew.add_profile(prof, nam, self.interp[nam])
        else:
            prof = getattr(self, profname1)-other
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)-other
                    profnew.add_profile(prof, nam, self.interp[nam])

        return profnew

    def __mul__(self, other):
        """
        Sums two profiles. They have to contain the same number of profiles. If they contain more than one profile the names have to be the same.
        """
        profname1 = self.names[0]

        if isinstance(other, AtmProfile):
            if len(self.names == 1):
                profname2 = other.names[0]
            else:
                profname2 = profname1

            if self.grid.grid.shape != other.grid.grid.shape:
                raise ValueError('Cannot sum two profiles with different grids.')

            prof = getattr(self, profname1)*getattr(other, profname2)
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)*getattr(other, nam)
                    profnew.add_profile(prof, nam, self.interp[nam])
        else:
            prof = getattr(self, profname1)*other
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)*other
                    profnew.add_profile(prof, nam, self.interp[nam])

        return profnew

    def __div__(self, other):
        """
        Sums two profiles. They have to contain the same number of profiles. If they contain more than one profile the names have to be the same.
        """
        profname1 = self.names[0]

        if isinstance(other, AtmProfile):
            if len(self.names == 1):
                profname2 = other.names[0]
            else:
                profname2 = profname1

            if self.grid.grid.shape != other.grid.grid.shape:
                raise ValueError('Cannot sum two profiles with different grids.')

            prof = getattr(self, profname1)/getattr(other, profname2)
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)/getattr(other, nam)
                    profnew.add_profile(prof, nam, self.interp[nam])
        else:
            prof = getattr(self, profname1)/other
            profnew = AtmProfile(self.grid, prof, profname1, self.interp[profname1])

            if len(self.names) > 1:
                for nam in self.names[1:]:
                    prof = getattr(self, nam)/other
                    profnew.add_profile(prof, nam, self.interp[nam])

        return profnew


    def calc(self, point, profname = None):
        """
        Interpolates the profile at the given point.
        :param point: np.array point to be considered. len(point) = self.ndim
        :param profname: name of the profile to calculate (es: 'temp', 'pres')
        :return: If only one profile is stored in AtmProfile, the output is a number. If there are more profiles stored in AtmProfile, the output of calc is a dict containing all interpolated values. If profname is set (ex. 'temp') returns the value of the profile 'temp' at the point.
        """
        try:
            len(point)
            if type(point) is list:
                point = np.array(point)
        except:
            point = np.array([point])

        if len(self.names) == 1 and profname is None:
            profname = self.names[0]

        if profname is not None:
            value = interp(getattr(self,profname), np.array(self.grid.grid), point, itype=self.interp[profname], hierarchy=self.grid.hierarchy)
            return value
        else:
            resu = dict()
            for nam in self.names:
                prof = getattr(self, nam)
                value = interp(prof, np.array(self.grid.grid), point, itype=self.interp[nam], hierarchy=self.grid.hierarchy)
                resu[nam] = value
            return resu


    def interp_copy(self, nomeprof, new_grid):
        """
        Interpolates the original profile nomeprof in the new_grid coords. Returns the interpolated profile.
        """

        print('funziona solo in 1D')
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


class AtmProfZeros(AtmProfile):
    def __init__(self, grid, profname, interp):
        prof = np.zeros(grid.grid[0].shape, dtype = float)
        AtmProfile.__init__(self, grid, prof, profname, interp)
        return


class TestAtmProf(AtmProfile):
    def __init__(self):
        alt = np.linspace(100.,500.,41)
        lat = np.linspace(20., 80., 7)
        grid = AtmGrid(['alt','lat'], [alt, lat])

        prof = np.linspace(0.,100.,41*7)
        prof = prof.reshape(41,7)
        AtmProfile.__init__(self, grid, prof, 'gigi', ['lin','box'])
        return


def prova_warning():
    gigi = np.zeros(10)
    pippo = gigi*2.0
    coso = gigi/pippo
    return coso


class AtmGridMask(AtmProfile):
    """
    Mask for discretization of the atmosphere. (e.g.) with respect to Bayesian Inversion Parameters.
    """
    def __init__(self, grid, mask, interp):
        AtmProfile.__init__(self, grid, mask, 'mask', interp)
        return

    def merge(self, other):
        """
        Merges two mask with different grids. The mask values are multiplied: mask(mio,tuo) = self.mask(mio)*other.mask(tuo)
        """
        nugrid = self.grid.merge(other.grid)
        dim1 = self.grid.ndim()
        dim2 = other.grid.ndim()

        nuprof = np.zeros(nugrid[0].shape)

        for point, num in zip(nugrid.points(), len(nuprof.flatten())):
            indx = np.unravel_index(num, nuprof.shape)
            nuprof[indx] = self.calc(point[:dim1])*other.calc(point[dim1:])

        interp = self.interp['mask']+other.interp['mask']

        numask = AtmGridMask(nugrid, nuprof, interp)

        return numask

#class Atmosphere(AtmProfile):


class PT_atm_AltLat(AtmProfile):
    """
    PT structure of the atmosphere discretized in alts and lats. linear (exp for pressure) interpolation in altitude, box in latitude.
    temps : list of vertical temp profiles, one for each latitude box
    press : list of vertical pres profiles, one for each latitude box
    """

    def __init__(self, temps, press, alt_grid, lat_grid):
        pass



#### FINE AtmProfile class

def interp(prof, grid, point, itype=None, hierarchy=None):
    """
    :param point: 1d array with ndim elements, the point to be considered
    :param grid: np.mgrid() array (ndim+1)
    :param prof: the profile to be interpolated (ndim array)
    :param itype: type of interpolation in each dimension, allowed values: 'lin', 'exp', 'box', '1cos'
    :param hierarchy: order for interpolation
    :other: dict with other infos on the grid.

    ---------------------- !!!!! IMPORTANT !!!!! ---------------------
    For 'box' interp, the grid values have to be set at the beginning of the box. (e.g.) if the lat limits are: (-90,-60), (-60,-30), ...
    the grid values will be: [-90, -60, -30, ...]
    ------------------------------------------------------------------
    :return:
    """

    try:
        ndim = len(point)
        point = np.array(point)
    except: # point è uno scalare
        ndim = 1
        point = np.array([point])

    if ndim != grid.ndim-1:
        raise ValueError('Point should have {:1d} dimensions, instead has {:1d}'.format(grid.ndim-1,ndim))

    if itype is None:
        itype = np.array(ndim*['lin'])
    if hierarchy is None:
        hierarchy = np.arange(ndim)

    # Calcolo i punti di griglia adiacenti, con i relativi pesi
    indxs = []
    weights = []
    for p, arr, ity in zip(point,grid,itype):
        indx, wei = find_between(np.unique(arr),p,ity)
        indxs.append(indx)
        weights.append(wei)

    # mi restringo l'array ad uno con solo i valori che mi interessano
    profi = prof
    for i in range(ndim):
        profi = profi.take(indxs[i],axis=i)

    # Calcolo i valori interpolati, seguendo hierarchy in ordine crescente. profint si restringe passo passo, perde un asse alla volta.
    hiesort = np.argsort(hierarchy)
    hiesort_dyn = []
    #print(hiesort)
    for num in range(ndim):
        ok = hiesort[0]
        hiesort_dyn.append(ok)
        hiesort = hiesort[1:]
        for nnn in range(len(hiesort)):
            if hiesort[nnn] > ok:
                hiesort[nnn] -= 1
    #print(hiesort_dyn)

    profint = profi
    for ax in hiesort_dyn:
        vals = [profint.take(ind,axis=ax) for ind in [0,1]]
        profint = int1d(vals,weights[ax],itype[ax])
        #print('aa',vals,weights[ax],itype[ax])
        #print('uuu',profint)

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


def round2SignifFigs(vals,n):
    """
    CREDITS: dmon on StackExchange

    (list, int) -> numpy array
    (numpy array, int) -> numpy array

    In: a list/array of values
    Out: array of values rounded to n significant figures

    Does not accept: inf, nan, complex

    >>> m = [0.0, -1.2366e22, 1.2544444e-15, 0.001222]
    >>> round2SignifFigs(m,2)
    array([  0.00e+00,  -1.24e+22,   1.25e-15,   1.22e-03])
    """

    n = n-1

    if np.all(np.isfinite(vals)) and np.all(np.isreal((vals))):
        eset = np.seterr(all='ignore')
        mags = 10.0**np.floor(np.log10(np.abs(vals)))  # omag's
        vals = np.around(vals/mags,n)*mags             # round(val/omag)*omag
        np.seterr(**eset)
        vals[np.where(np.isnan(vals))] = 0.0           # 0.0 -> nan -> 0.0
    else:
        raise IOError('Input must be real and finite')
    return vals


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

def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

@counted
def stocazzo():
    print('stocazzo')


def find_between(array, value, interp='lin', thres = 1.e-5):
    """
    Returns the indexes of the two closest points in array and the relative weights for interpolation.
    :param array: grid Array
    :param value: value requested
    :param interp: type of weighting ('lin' -> linear interpolation,'exp' -> exponential interpolation,
    'box' -> nearest neighbour,'1cos' -> 1/cos(x) for SZA values)
    :return:
    """

    array = np.array(array)
    if interp == 'lin' or interp == 'exp':
        valo = value
        mino = np.min(array)
        maxo = np.max(array)
        thres = thres*(array[1]-array[0])
        if valo < mino-thres or valo > maxo+thres:
            raise ValueError('Extrapolation required! val: {} min: {} max: {}'.format(value, np.min(array), np.max(array)))
        elif valo < mino:
            #raise RuntimeWarning('ATTENTION! slight mismatch in max/min grid values and point required. {} {} diff {}'.format(value, np.min(array), value-np.min(array)))
            idx = np.argsort(array)[:2]
            weights = np.array([1,0])
        elif  valo > maxo:
            #raise RuntimeWarning('ATTENTION! slight mismatch in max/min grid values and point required. {} {} diff {}'.format(value, np.min(array), value-np.min(array)))
            idx = np.argsort(array)[-2:]
            weights = np.array([0,1])
        else:
            ndim = array.ndim
            dists = np.sort(np.abs(array-value))
            indxs = np.argsort(np.abs(array-value))
            vals = array[indxs][:2]
            idx = indxs[:2]
            dis = dists[:2]
            weights = np.array(weight(value,vals[0],vals[1],itype=interp))
    elif interp == 'box':
        thres = abs(thres*(array[1]-array[0]))
        idx1 = np.argwhere(array < value+thres)[-1]
        idx = [idx1, idx1]
        weights = [1,0]

    return idx, weights


def weight(p,pg1,pg2,itype='lin'):
    """
    Weight of pg1 and pg2 in calculating the interpolated value p, according to type.
    :param pg1: Point 1 in the grig
    :param pg2: Point 2 in the grid
    :param itype: 'lin' : (1-difference)/step; 'exp' : (1-log difference)/log step; 'box' : 1 first, 0 the other; '1cos' : (1-1cos_diff)/1cos_step
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
        w1 = 1
        w2 = 0
        # THIS IS FOR EQUALLY SPACED BOXES
        # if abs(p-pg1) < abs(p-pg2):
        #     w1=1
        #     w2=0
        # else:
        #     w1=0
        #     w2=1
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


def trova_spip(ifile, hasha = '#', read_past = False):
    """
    Trova il '#' nei file .dat
    """
    gigi = 'a'
    while gigi != hasha :
        linea = ifile.readline()
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
    freq = np.array([float(r) for r in data_arr[:, 0]])
    obs = data_arr[:, 1:2*n_limb+2:2]
    obs = obs.astype(float)
    flags = data_arr[:, 2:2*n_limb+2:2]
    flags = flags.astype(int)

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


def read_input_prof_gbb(filename, ptype, n_alt_max =None, n_alt = 151, alt_step = 10.0, n_gas = 86, n_lat = 4, read_bad_names = False):
    """
    Reads input profiles from gbb standard formatted files (in_temp.dat, in_pres.dat, in_vmr_prof.dat).
    Profile order is from surface to TOA.
    type = 'vmr', 'temp', 'pres'
    :return: profiles
    read_bad_names is to read profiles of stuff that is not in the HITRAN list.
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'r')

    if(ptype == 'vmr'):
        print(ptype)
        trova_spip(infile)
        trova_spip(infile)
        proftot = []
        mol_names = []
        for i in range(n_gas):
            try:
                lin = infile.readline()
                print(lin)
                num = int(lin.split()[0])
                nome = lin.split()[1]
            except:
                break

            try:
                nome = find_molec_metadata(num, 1)['mol_name']
            except:
                if not read_bad_names:
                    break

            prof = []
            while len(prof) < n_alt:
                line = infile.readline()
                prof += list(map(float, line.split()))
            prof = np.array(prof[::-1])*1.e-6 # SETTING UNITY TO ABSOLUTE FRACTION, NOT PPM
            if n_alt_max is not None and n_alt_max < n_alt:
                prof = prof[:n_alt_max]
            proftot.append(prof)
            try:
                mol_names.append(find_molec_metadata(i+1, 1)['mol_name'])
            except:
                mol_names.append(nome)

            for j in range(n_lat-1): # to skip other latitudes
                prof = []
                while len(prof) < n_alt:
                    line = infile.readline()
                    prof += list(map(float, line.split()))

        proftot = dict(zip(mol_names,proftot))

    if(ptype == 'temp' or ptype == 'pres'):
        print(ptype)
        trova_spip(infile)
        trova_spip(infile)
        prof = []
        while len(prof) < n_alt:
            line = infile.readline()
            prof += list(map(float, line.split()))
        proftot = np.array(prof[::-1])
        if n_alt_max is not None and n_alt_max < n_alt:
            proftot = proftot[:n_alt_max]

    return proftot


def write_input_prof_gbb(filename, prof, ptype, n_alt = 151, alt_step = 10.0, nlat = 4, descr = '', script=__file__):
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

    if(ptype == 'vmr'):
        strin = '{:10.3e}'
        infile.write('VMR of molecules (ppmV)\n')
    elif(ptype == 'temp'):
        strin = '{:10.5f}'
        infile.write('Temperature (K)\n')
    elif(ptype == 'pres'):
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
            # prof = AtmProfile(prof, alts, profname = 'vibtemp')
            alt_gri = AtmGrid('alt', alts)
            prof = AtmProfile(alt_gri, prof, 'vibtemp', 'lin')

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


def extract_quanta_HITRAN(mol, iso, lev_string):
    """
    Extracts from string the vibrational quantum numbers and possibly simmetry/parity or other quantum numbers. Made on the HITRAN 2004-2012 format. Could change with future versions and with other databases.
    """
    groups = [[5,14,15,16,17,22,36]]
    groups.append([7])
    groups.append([8,13,18])
    groups.append([4,19,23])
    groups.append([2])
    groups.append([1,3,9,10,21,31,33,37])
    groups.append([26])
    groups.append([11,28])
    groups.append([20,25,29])
    groups.append([6])
    groups.append([24,27,12,30,32,35,38,39])

    num = 0
    for gru in groups:
        num += 1
        if mol in gru:
            ok = True
            break

    if not ok:
        raise ValueError('Molecule {} not found in the list!!'.format(mol))

    if mol == 6 and iso == 3:
        num = 11

    others = None
    vib_quanta = None

    try:
        if num == 1:
            vib_quanta = [int(lev_string)]
            minimal_string = lev_string.strip()
        elif num == 2:
            minimal_string = lev_string.strip()
            vib_quanta = [int(lev_string[13:])]
            electronic_level = lev_string[12]
            others = dict()
            others['electronic_level'] = electronic_level
        elif num == 3:
            minimal_string = lev_string.strip()
            vib_quanta = [int(lev_string[13:])]
            electronic_level = lev_string[7:11]
            others = dict()
            others['electronic_level'] = electronic_level
        elif num == 4:
            minimal_string = lev_string.strip()
            vib_quanta = map(int,lev_string.split())
        elif num == 5:
            minimal_string = lev_string[:-1].strip()
            vib_quanta = map(int,lev_string[:-1].split())
            fermi_res = lev_string[-1]
            others = dict()
            others['fermi_res'] = fermi_res
        elif num == 6:
            minimal_string = lev_string.strip()
            vib_quanta = map(int,lev_string.split())
        elif num == 7:
            minimal_string = lev_string[:12].strip()
            vib_quanta = map(int,lev_string[:12].split())
            others = dict()
            others['parity'] = lev_string[12]
            others['fermi_res'] = lev_string[13]
            others['S-field parity'] = lev_string[-1]
        elif num == 8:
            minimal_string = lev_string[:13].strip()
            vib_quanta = map(int,lev_string[:13].split())
            others = dict()
            others['simmetry'] = lev_string[13:]
        elif num == 9:
            minimal_string = lev_string.strip()
            vib_quanta = map(int,lev_string.split())
        elif num == 10:
            minimal_string = lev_string[:11].strip()
            vib_quanta = map(int,lev_string[:11].split())
            others = dict()
            others['simmetry'] = lev_string[11:]
        elif num == 11:
            minimal_string = lev_string.strip()
    except Exception as cazzillo:
        print('Exception found... for level string {}'.format(lev_string))
        print(cazzillo)
        return '', None, None

    return minimal_string, vib_quanta, others


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
                if len(_lev_.simmetry) == 0:
                    print('Skipping level '+lev)
                    continue
                if len(_lev_.simmetry[0]) != 15:
                    errstring = 'Length of level string is {} instead of 15 for level {} of molecule {}, stopping....'.format(len(_lev_.lev_string),lev,iso)
                    raise ValueError(errstring)
                infile.write('Hitran Code of level, internal code of level, number of simmetries, multiplier(depr.), Level Energy:\n')
                infile.write('{:1s}\n'.format('#'))
                print(_lev_.simmetry)
                infile.write('{:15.15s}{:5d}{:5d}{:20.3f}{:12.4f}\n'.format(_lev_.simmetry[0],ioo,len(_lev_.simmetry)-1,1.0,_lev_.energy))

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


def add_nLTE_molecs_from_tvibmanuel(planet, filename, n_alt_max = None, linee = None, add_fundamental = True, extend_to_alt = None):
    """
    Returns a set of molecs with the correct levels and tvibs.
    """
    alts, mol_names, levels, energies, vib_ok = read_tvib_manuel(filename, n_alt_max = n_alt_max, extend_to_alt = extend_to_alt)


    molecs = dict()
    T_kin = planet.atmosphere.temp
    alt_grid = AtmGrid('alt', planet.atmosphere.grid.coords['alt'])
    T_kin = AtmProfile(alt_grid, T_kin, 'vibtemp', 'lin')

    for molcoso, lev, ene, vib in zip(mol_names, levels, energies, vib_ok):
        mol = int(molcoso[:-1])
        iso = int(molcoso[-1])

        info = find_molec_metadata(mol, iso)

        try:
            molec = molecs[info['mol_name']]
        except:
            molecs[info['mol_name']] = Molec(mol, name = info['mol_name'])
            molec = molecs[info['mol_name']]

        striso = 'iso_{:1d}'.format(iso)

        try:
            isomol = getattr(molec, striso)
        except:
            molec.add_iso(iso, ratio = info['iso_ratio'], LTE = False)
            isomol = getattr(molec, striso)

            # add_fundamental
            if add_fundamental:
                minstr = extract_quanta_HITRAN(mol, iso, lev)[0]
                lev_0 = ''
                for lett in minstr:
                    try:
                        num = int(lett)
                        lev_0 += '0'
                    except:
                        lev_0 += lett
                isomol.add_level(lev_0, 0.0, vibtemp = T_kin)

        isomol.add_level(lev, ene, vibtemp = vib)

    if linee is not None:
        for molec in molecs.values():
            for iso in molec.all_iso:
                getattr(molec, iso).add_simmetries_levels(linee)

    return molecs


def read_tvib_manuel(filename, n_alt_max = None, extend_to_alt = None):
    """
    Reads input atmosphere in manuel standard.
    :param filename:
    :return:
    """

    infile = open(filename,'r')
    trova_spip(infile,hasha='$')
    n_alt = int(infile.readline())
    if n_alt_max is not None and n_alt > n_alt_max:
        n_alt = n_alt_max
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
        ind_e = linea.index(linea.split()[-1])
        levels.append(linea[0:ind_e].strip())
        energies.append(float(linea.split()[-1]))
        molecs.append(infile.readline().rstrip())
        infile.readline()

        prof = []
        while len(prof) < n_alt:
            line = infile.readline()
            prof += list(map(float, line.split()))
        pres = np.array(prof)
        vibtemps.append(pres)

    infile.close()

    vib_ok = []
    alts = np.array(alts)
    alt_grid = AtmGrid('alt', alts)
    for tempu in vibtemps:
        if extend_to_alt is not None:
            if extend_to_alt > max(alts):
                stp = alts[1]-alts[0]
                alts2 = np.arange(max(alts)+stp, extend_to_alt+stp/2, stp)
                alts_ok = np.append(alts, alts2)
                vals = np.array(len(alts2)*[tempu[-1]])
                tempu = np.array(tempu)
                tempu = np.append(tempu, vals)
                alt_grid = AtmGrid('alt', alts_ok)
        tempu_ok = AtmProfile(alt_grid, tempu, 'vibtemp', 'lin')
        vib_ok.append(tempu_ok)

    return alts, molecs, levels, energies, vib_ok


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
        # Skips commented lines:
        lines = [line for line in lines if not line.lstrip()[:1] == '#']

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


def read_bands(filename, wn_range = None):
    """
    Reads bands and ILS fwhm of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    wl = np.array([float(r) for r in data_arr[:, 0]])
    sig = np.array([float(r) for r in data_arr[:, 3]])
    if wn_range is not None:
        cond = (wl >= wn_range[0]) & (wl <= wn_range[1])
        wl = wl[cond]
        sig = sig[cond]
    infile.close()

    spgri = spcl.SpectralGrid(wl)
    bands = spcl.SpectralObject(sig, spgri)

    return bands

def read_noise(filename, wn_range = None):
    """
    Reads noise of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    wl = np.array([float(r) for r in data_arr[:, 0]])
    err = np.array([float(r) for r in data_arr[:, 1]])
    if wn_range is not None:
        cond = (wl >= wn_range[0]) & (wl <= wn_range[1])
        wl = wl[cond]
        err = err[cond]
    infile.close()

    spgri = spcl.SpectralGrid(np.array(wl))
    err = spcl.SpectralObject(np.array(err), spgri)

    return err

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
