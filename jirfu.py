#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
import matplotlib.lines as lin
import math as m
from numpy import linalg as LA

print(sys.path, len(sys.path))
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import pickle
import spect_base_module as sbm
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

class JirPix(sbm.Pixel):
    """
    Pixel for Jiram.
    """
    def __init__(self, keys = None, things = None):
        if keys is None:
            return
        for key, thing in zip(keys,things):
            setattr(self,key,thing)
        self.spe = 1e-3*self.spe
        fondo = fondojir(self.wl, self.spe)
        mask = findspi(self.wl,self.spe)
        self.mask = mask
        int_L = self.integr(range=[3200,3700],fondo=fondo)
        self.int_L = int_L
        return

    def plotsim(self):
        pl.plot(self.wlsim,self.obs,label='Obs')
        pl.plot(self.wlsim,self.sim,label='Sim')
        pl.plot(self.wlsim,self.resid,label='Resid')
        pl.grid()
        pl.legend()
        pl.show()


class JirSet(sbm.PixelSet):
    """
    Set for Jiram.
    """
    def __new__(cls, input_array, descr=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        print(cls)
        obj = np.asarray(input_array).view(cls)
        print(type(obj))
        # add the new attribute to the created instance
        obj.descr = descr
        fit_results = ['h3p_col','err_col','h3p_temp','err_temp','chisq','offset','ch4_col','err_ch4','wl_shift']
        for name in fit_results:
            setattr(obj,name,None)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: pass
        fit_results = ['h3p_col','err_col','h3p_temp','err_temp','chisq','offset','ch4_col','err_ch4','wl_shift']
        for name in fit_results:
            lui = getattr(obj,name,None)
            setattr(self,name,lui)
        self.descr = getattr(obj, 'descr', None)
        for name in obj[0].__dict__.keys():
            try:
                setattr(self,name,np.array([getattr(pix,name,None) for pix in self]))
            except:
                setattr(self,name+'_ok',np.array([getattr(pix,name,None) for pix in self]))
        return

    def stereomap(self, attr, **kwargs):
        attr_to_plot = getattr(self,attr)
        stereomap2(self.pc_lat,self.pc_lon,attr_to_plot,**kwargs)
        return

    def stereoplot(self, attr, **kwargs):
        attr_to_plot = getattr(self,attr)
        stereoplot(self.pc_lat,self.pc_lon,attr_to_plot,**kwargs)
        return

    def scatter(self, attr, **kwargs):
        attr_to_plot = getattr(self,attr)
        stereoplot(self.pc_lat,self.pc_lon,attr_to_plot,**kwargs)
        return

    def read_res(self, cart='', lwlsh = False, lch4 = True):
        """
        Leggi i risultati da cart e appiccicali ai pixel. Crea nuovi attributi anche per PixelSet.
        :param cart:
        :return:
        """
        fit_results = ['h3p_col','err_col','h3p_temp','err_temp','chisq','offset','ch4_col','err_ch4','wl_shift']
        if not lwlsh:
            fit_results = fit_results[:-1]
        if not lch4:
            fit_results = fit_results[:-3]

        ii,col1, err_col = read_res_jir_3(cart+'CD-H3p.dat')
        iit,temp1, err_temp = read_res_jir_3(cart+'VT-H3p.dat')
        iic,chi1 = read_res_jir_4(cart+'chisq.dat')
        iio,off1 = read_res_jir_4(cart+'offset_ok.dat')
        if lwlsh: iis,shi1 = read_res_jir_4(cart+'shift.dat')
        if lch4: ii4,ch41, err_ch41 = read_res_jir_3(cart+'CD-CH4.dat')

        col = np.zeros(len(self))
        err_c = np.zeros(len(self))
        temp = np.zeros(len(self))
        err_t = np.zeros(len(self))
        chi = np.zeros(len(self))
        off = np.zeros(len(self))
        if lwlsh: shi = np.zeros(len(self))
        if lch4:
            ch4 = np.zeros(len(self))
            err_ch4 = np.zeros(len(self))

        print(len(self), len(col1))
        col[ii] = col1
        err_c[ii] = err_col
        temp[iit] = temp1
        err_t[iit] = err_temp
        chi[iic] = chi1
        chi[chi == 0.0] = np.nan
        off[iio] = off1
        if lwlsh: shi[iis] = shi1
        if lch4:
            ch4[ii4] = ch41
            err_ch4[ii4] = err_ch41

        results = [col,err_c,temp,err_t,chi,off]
        if lch4:
            results.append(ch4)
            results.append(err_ch4)
        if lwlsh: results.append(shi)

        for attr,res in zip(fit_results,results):
            for pix,res1 in zip(self,res):
                setattr(pix,attr,res1)
            setattr(self,attr,res)

        return

    def set_res(self,results,fit_results=None,lwlsh=False,lch4=True):
        """
        Sets the values of the results, already read elsewhere.
        """
        if fit_results is None:
            fit_results = ['h3p_col','err_col','h3p_temp','err_temp','chisq','offset','ch4_col','err_ch4','wl_shift']
            if not lwlsh:
                fit_results = fit_results[:-1]
            if not lch4:
                fit_results = fit_results[:-3]
            print('Warning! Result names set to default: {}'.format(fit_results))

        for attr,res in zip(fit_results,results):
            # for pix,res1 in zip(self,res):
            #     setattr(pix,attr,res1)
            setattr(self,attr,res)

        return


###########################################


# def convert_to_pix(array):
#     """
#     Converts Jiram structured array pixs to sbm.SetPixel array. Reads the results from folder cart.
#     :param array:
#     :param cart:
#     :return:
#     """
def fmt_10(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def fmt(lon):
    if rcParams['text.usetex']:
        lonlabstr = r'${%s\/^{\circ}}$'%lon
    else:
        lonlabstr = u'%s\N{DEGREE SIGN}'%lon
    return lonlabstr

def stereoplot(lon,lat,col,nomefi=None,live = False,polo='N',minu=0,maxu=0,title='',show=False,aur_model='stat',
               proj='orto',condpo=None, cbarform=None, cbarmult = None, cbarlabel = '', step = None, pdf=None,
               edges=None,invert_cmap=False,cmap=None, axu=None, alpha=0.4, blats=None, style = 'x-large'):
    """
    Plots points on a stereographic map with colors.
    :return:
    """
    print('FILE : {}'.format(nomefi))

    if cmap is None:
        cmap = 'jet'
    if invert_cmap: cmap = cmap+'_r'

    rsp = [71492.,66854.]
    if axu is None:
        fig = pl.figure(figsize=(8, 6), dpi=72)
        ax = pl.subplot()
    else:
        ax = pl.subplot(axu)

# Valid font size are large, medium, smaller, small, x-large, xx-small, larger, x-small, xx-large, None
    if style == 'medium':
        fnsz = 10
    elif style == 'small':
        fnsz = 7
    elif style == 'large':
        fnsz = 13
    elif style == 'x-large':
        fnsz = 16
    else:
        fnsz = 10
        style = 'medium'

    pl.title(title, y=1.05,fontsize=style)

    if polo == 'N':
        if blats is None:
            blats = np.array([68,50,68,80]) # order in blats is right, up, left, down
        blats = np.array(blats)
        signs = np.array([1,1,-1,-1])
        xmis = signs*rsp[0]*np.cos(np.pi*blats/180.0)
        # xmi_ur = rsp[0]*np.cos(np.pi*blats[0]/180.0)
        # ymi_ur = rsp[0]*np.cos(np.pi*blats[1]/180.0)
        # xmi_ll = -rsp[0]*np.cos(np.pi*blats[2]/180.0)
        # ymi_ll = -rsp[0]*np.cos(np.pi*blats[3]/180.0)
        if proj == 'orto': map = Basemap(projection='ortho',lon_0=0,lat_0=90,resolution='l',rsphere=rsp[0],
                                        llcrnrx=xmis[2],llcrnry=xmis[3],urcrnrx=xmis[0],urcrnry=xmis[1])
        if proj == 'stereo': map = Basemap(projection='npstere',boundinglat=np.min(blats),lon_0=0,lat_0=90,resolution='l')
        map.drawparallels(np.arange(0,90,10))
    else:
        if blats is None:
            blats = np.array([-60,-60,-60,-70]) # order in blats is right, up, left, down
        blats = np.array(blats)
        signs = np.array([1,1,-1,-1])
        xmis = signs*rsp[0]*np.cos(np.pi*blats/180.0)
        if proj == 'orto': map = Basemap(projection='ortho',lon_0=0,lat_0=-90,resolution='l',rsphere=rsp[0],
                                llcrnrx=xmis[2],llcrnry=xmis[3],urcrnrx=xmis[0],urcrnry=xmis[1])
        if proj == 'stereo': map = Basemap(projection='spstere',boundinglat=np.max(blats),lon_0=180,lat_0=-90,resolution='l')
        map.drawparallels(np.arange(-80,0,10))

    stat_x,stat_y,vip4_lat,vip4_lon = leggi_map_aur(polo)
    if polo == 'N':
        x0, y0 = map(0.,90.)
    else:
        x0, y0 = map(0.,-90.)
    stat_x = stat_x+x0
    stat_y = stat_y+y0

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,1,1],fontsize=fnsz,fmt=fmt)

    x, y = map(lon,lat)

    if condpo is None and edges is None:
        sca = map.scatter(x,y,c = col,s=5,edgecolors='none',vmin=minu,vmax=maxu,cmap=cmap)
    elif edges is None:
        sca = map.scatter(x[~condpo],y[~condpo],s=2,color='grey',edgecolors='none',cmap=cmap)
        sca = map.scatter(x[condpo],y[condpo],c=col[condpo],s=5,edgecolors='none',vmin=minu,vmax=maxu,cmap=cmap)
    else:
        edges_xy = []
        for lonlats in edges:
            edges_xy.append([map(lo,la) for [lo,la] in lonlats])


        if maxu == 0:
            macs = np.max(col[(~np.isnan(col))])
            mins = np.min(col[(~np.isnan(col))])
        else:
            macs = maxu
            mins = minu
        cma = getattr(cm,cmap)

        pixies, edgies = pixel_proj(edges_xy,alpha=alpha,facecolors=(col-mins)/(macs-mins),edgecolors=(col-mins)/(macs-mins),cmap=cmap,lw=0.2)
        #pixies.set_array(col)
        #pixies.set_clim([mins,macs])

        ax.add_collection(pixies)
        ax.add_collection(edgies)
        sca = pl.cm.ScalarMappable(cmap=cmap, norm=pl.Normalize(vmin=mins, vmax=macs))
        sca.set_array(col)

    if cbarmult is not None:
        cbarlabel = cbarlabel.format(cbarmult)

    if cbarform is not None:
        cb = pl.colorbar(sca, format=cbarform, pad = 0.1, shrink = 0.8, aspect = 15)
        cb.set_label(cbarlabel,fontsize=style)
    else:
        cb = pl.colorbar(sca, pad = 0.1, shrink = 0.8, aspect = 15)
        cb.set_label(cbarlabel,fontsize=style)

    if step is not None:
        step = 2*step
        ticks = np.arange(minu,maxu+step/10.,step)
        if cbarmult is not None and cbarform is not None:
            labels = [('{:'+cbarform[1:]+'}').format(ti/10**cbarmult) for ti in ticks]
        elif cbarmult is None and cbarform is not None:
            labels = [('{:'+cbarform[1:]+'}').format(ti) for ti in ticks]
        else:
            labels = ['{}'.format(ti) for ti in ticks]
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
        cb.set_label(cbarlabel,fontsize=style)

    cb.ax.tick_params(labelsize=fnsz)

    vip4x, vip4y = map(360-vip4_lon,vip4_lat)

    vip4x = np.append(vip4x,vip4x[0])
    vip4y = np.append(vip4y,vip4y[0])
    stat_x = np.append(stat_x,stat_x[0])
    stat_y = np.append(stat_y,stat_y[0])
    aur_lw = 2.0
    if aur_model == 'VIP4':
        pl.plot(vip4x,vip4y,color='white',linewidth=aur_lw)
        pl.plot(vip4x,vip4y,color='black',linewidth=aur_lw,linestyle='--')
    elif aur_model == 'stat':
        pl.plot(stat_x,stat_y,color='white',linewidth=aur_lw)
        pl.plot(stat_x,stat_y,color='black',linewidth=aur_lw,linestyle='-')
    else:
        pl.plot(vip4x,vip4y,color='white',linewidth=aur_lw)
        pl.plot(vip4x,vip4y,color='black',linewidth=aur_lw,linestyle='--')
        pl.plot(stat_x,stat_y,color='white',linewidth=aur_lw)
        pl.plot(stat_x,stat_y,color='black',linewidth=aur_lw,linestyle='-')

    if live: show=True
    if show: pl.show()
    if not live and axu is None:
        if nomefi is not None:
            lol = nomefi.find('.')
            form = nomefi[lol+1:]
            print('DPI:',fig.dpi)
            fig.savefig(nomefi, format=form, dpi=fig.dpi)
        if pdf is not None: pdf.savefig(fig)
        pl.close()

    return


def pixel_proj(coords,alpha=0.5,facecolors=None,edgecolors=None,do_norm=False,cmap='jet',lw=0.2):
    """
    From the list of the pixel edges returns a PatchCollection to plot on the map. Optionally the color of the color of the pixels can be set already.
    :param coords: array 2 x 4 x ndim -> [[[x11,y11],[x12,y12],...] .. [[xn1,yn1], ... ] .. ]
    :param facecolors: array ndim setting the facecolor (is normalized from 0 to 1)
    :param edgecolors: array ndim setting the edgecolor (is normalized from 0 to 1)
    :param alpha: transparency of the color
    :param cmap: name of the color map
    :param do_norm: if set face and edgecolors are normalized before being passed to cmap
    :return: pixs,edgs : two PatchCollection s for the inner pixel and the borders
    """

    import matplotlib.cm as cm
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    patches = []
    patches2 = []

    #print(edgecolors,'uauguaguua')
    #print(np.sum(np.isnan(edgecolors)))
    if do_norm and edgecolors is not None and facecolors is not None:
        edgecolors = normalize_nonan(edgecolors)
        facecolors = normalize_nonan(facecolors)
    #print(edgecolors,'Uauau',np.min(edgecolors[~np.isnan(edgecolors)]),np.max(edgecolors[~np.isnan(edgecolors)]))
    if edgecolors is None:
        edgecolors = 'none'

    for pixels in coords:
        vertici = [zi for zi in pixels]
        polygon = Polygon(np.array(vertici))# fill = True, alpha=alpha, facecolor=fac, edgecolor='none')
        border = Polygon(np.array(vertici))# fill=False, edgecolor=ed, alpha=1.0)
        patches.append(polygon)
        patches2.append(border)

    cma = getattr(cm,cmap)

    pixs = PatchCollection(patches)
    edgs = PatchCollection(patches2)

    pixs.set_alpha(0.5)
    pixs.set_facecolors(cma(facecolors))
    pixs.set_edgecolor('none')
    edgs.set_facecolor('none')
    edgs.set_edgecolors(cma(edgecolors))
    edgs.set_lw(lw)
    #print(edgs.get_edgecolors(),'UAUA\n',edgs.get_facecolors(),'UAUA\n')
    #print(pixs.get_edgecolors(),'UAUA\n',pixs.get_facecolors(),'UAUA\n')

    # if color is not None:
    #     p.set_array(np.array(color))

    return pixs, edgs

def normalize_nonan(vec):
    mins = np.min(vec[~np.isnan(vec)])
    maxs = np.max(vec[~np.isnan(vec)])
    print('Estremi: ',mins,maxs)
    vec = (vec-mins)/(maxs-mins)
    return vec


def stereomap2(lon,lat,col,nomefi=None,polo='N',proj = 'orto', minu=0, maxu=0, title='', show=False, lonlat=False, xres=50, lonres=180, latres=30, ncont=15, step=None, salta=2, cbarform=None, cbarlabel = '', cbarmult = None, addpoints=False, condpo=None, minnum=2, image=False, interp='nearest', npoints=False, aur_model='stat', live=False, pdf=None, axu = None, blats=None, cmap='jet', style = 'large', normalize = False, print_values = False):
    """
    Plots points on an ortho- or a stereographic map with colors.
    :return:
    """
    from matplotlib.colors import LogNorm

    print('FILE : {}'.format(nomefi))

    rsp = [71492.,66854.]
    if axu is None:
        fig = pl.figure(figsize=(8, 6), dpi=150)
        ax = pl.subplot()
    else:
        ax = pl.subplot(axu)

    # Valid font size are large, medium, smaller, small, x-large, xx-small, larger, x-small, xx-large, None
    if style == 'medium':
        fnsz = 10
    elif style == 'small':
        fnsz = 7
    elif style == 'large':
        fnsz = 13
    else:
        fnsz = 10
        style = 'medium'

    pl.title(title, y=1.05,fontsize=style)

    if polo == 'N':
        if blats is None:
            blats = np.array([68,50,68,80]) # order in blats is right, up, left, down
        blats=np.array(blats)
        signs = np.array([1,1,-1,-1])
        xmis = signs*rsp[0]*np.cos(np.pi*blats/180.0)
        print('XMIS',xmis)
        if proj == 'orto': map = Basemap(projection='ortho',lon_0=0,lat_0=90,resolution='l',rsphere=rsp[0],
                                        llcrnrx=xmis[2],llcrnry=xmis[3],urcrnrx=xmis[0],urcrnry=xmis[1])
        if proj == 'stereo': map = Basemap(projection='npstere',boundinglat=np.min(blats),lon_0=0,lat_0=90,resolution='l')
        map.drawparallels(np.arange(0,90,10))
    else:
        if blats is None:
            blats = np.array([-60,-60,-60,-70]) # order in blats is right, up, left, down
        blats=np.array(blats)
        signs = np.array([1,1,-1,-1])
        xmis = signs*rsp[0]*np.cos(np.pi*blats/180.0)
        if proj == 'orto': map = Basemap(projection='ortho',lon_0=0,lat_0=-90,resolution='l',rsphere=rsp[0],
                                llcrnrx=xmis[2],llcrnry=xmis[3],urcrnrx=xmis[0],urcrnry=xmis[1])
        if proj == 'stereo': map = Basemap(projection='spstere',boundinglat=np.max(blats),lon_0=180,lat_0=-90,resolution='l')
        map.drawparallels(np.arange(-80,0,10))

    stat_x,stat_y,vip4_lat,vip4_lon = leggi_map_aur(polo)
    if polo == 'N':
        x0, y0 = map(0.,90.)
        blob = np.min(blats)
    else:
        x0, y0 = map(0.,-90.)
        blob = np.max(blats)
    stat_x = stat_x+x0
    stat_y = stat_y+y0

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,1,1],fontsize=fnsz,fmt=fmt)

    # CHECK if there are nans in the coordinates or in the data:
    zoi = (~np.isnan(lon)) & (~np.isnan(lat)) & (~np.isnan(col))
    lon = lon[zoi]
    lat = lat[zoi]
    col = col[zoi]

    x, y = map(lon,lat)
    #print(np.min(x),np.min(y),np.max(x),np.max(y))
    print('CENTRO: ',[x0,y0])

    # print(map(0.,-60.),map(90.,-60.),map(180.,-60.),map(270.,-60.))

    # Trovo i minimi e massimi delle coordinate del quadratone con massima estensione nel plot
    xlim= x0 + np.array([-np.max(np.abs(xmis)),np.max(np.abs(xmis))])
    ylim= y0 + np.array([-np.max(np.abs(xmis)),np.max(np.abs(xmis))])
    nx=xres
    ny=xres
    nstep=xres*1j
    xgri, ygri = np.mgrid[xlim[0]:xlim[1]:nstep, ylim[0]:ylim[1]:nstep]
    xstep = xgri[1,0]-xgri[0,0]
    ystep = ygri[0,1]-ygri[0,0]

    print('AREA PIXEL: {:9.4e} km^2. Step X: {:8.3e} km, Step Y: {:8.3e} km.\n'.format(xstep*ystep,xstep,ystep))

    nsteplot=lonres*1j
    nsteplat=latres*1j
    if polo == 'N':
        grid_lat, grid_lon = np.mgrid[np.min(blats):90:nsteplat, 0:360:nsteplot]
    else:
        grid_lat, grid_lon = np.mgrid[-90:np.max(blats):nsteplat, 0:360:nsteplot]
    xg, yg = map(grid_lon, grid_lat)

    nlat = latres #np.shape(grid_lat)[0]
    nlon = lonres #np.shape(grid_lat)[1]
    steplo = 360/nlon
    stepla = 30/nlat

    # tolgo i NaN dai vettori
    cond2 = (~np.isnan(col))
    # print('CONTROLLO dentro stereomap2',nomefi,len(col),len(lon),len(lat),len(cond2),len(x),len(y))
    col2 = col[cond2]
    lon2 = lon[cond2]
    lat2 = lat[cond2]
    x2 = x[cond2]
    y2 = y[cond2]

    #grid_near = griddata((x,y), col, (xg, yg), method='nearest')
    #grid_near = griddata((lon,lat), col, (grid_lon, grid_lat), method='nearest')

    # Commento: i dati vanno grigliati a mano. Per ogni punto prendo quelli più vicini e li medio, buttando via i nan.

    lonlat=False

    if(lonlat):
        cols = -np.ones((nlat,nlon))
        num = np.zeros((nlat,nlon))
        for i in range(nlat):
            for j in range(nlon):
                glo = grid_lon[i,j]
                gla = grid_lat[i,j]
                cond = (lon2-glo >= -steplo/2) & (lon2-glo < steplo/2) & (lat2-gla >= -stepla/2) & (lat2-gla < stepla/2)
                if len(col2[cond]) > 0:
                    cols[i,j] = np.mean(col2[cond])
                num[i,j] = len(col2[cond])
    else:
        cols = -np.ones((nx,ny))
        num = np.zeros((nx,ny))
        steplo = xgri[1,0]-xgri[0,0]
        stepla = ygri[0,1]-ygri[0,0]
        for i in range(nx):
            for j in range(ny):
                glo = xgri[i,j]
                gla = ygri[i,j]
                cond = (x2-glo >= -steplo/2) & (x2-glo < steplo/2) & (y2-gla >= -stepla/2) & (y2-gla < stepla/2)
                #print(i,j,np.sum(cond),np.min(x2),np.max(x2),glo,np.min(y2),np.max(y2),gla)
                if len(col2[cond]) > 0:
                    cols[i,j] = np.mean(col2[cond])
                num[i,j] = len(col2[cond])

    cols[(cols<0)]=float(np.nan)
    cols[(num < minnum)]=float(np.nan)
    #cols = np.ma.masked_array(cols, mask=cols<0.0)

    if print_values:
        filosname = nomefi[:nomefi.index('.')]+'.dat'
        filos = open(filosname,'w')
        filos.write('#{:7s}{:8s}{:12s}\n'.format('Lat','Lon','Value'))
        for xxi,yyi,co in zip(xgri.flatten(),ygri.flatten(),cols.flatten()):
            lo, la = map(xxi, yyi, inverse=True)
            if abs(la) < np.min(abs(blats)): continue
            try:
                filos.write('{:8.2f}  {:8.2f}  {:12.3f}\n'.format(la,lo,co))
            except Exception as cazzillo:
                print(cazzillo)
                print(type(la),type(lo),type(co))
                print('PROBLEMA, converto masked a nan: {} {} {}'.format(la,lo,co))
                filos.write('{:8.2f}  {:8.2f}  {:12.3f}\n'.format(la,lo,np.nan))
        filos.close()
    #pl.pcolormesh(xg, yg, cols)

    #cols[20,40]=np.nan
    conan = np.isnan(cols)
    cols = np.ma.MaskedArray(cols,conan)
    print(np.sum(cols > 0))
    print(cols)

    if step is not None:
        ncont = np.ceil((maxu-minu)/step)
        maxu = minu+ncont*step

    if lonlat:
        if(minu == maxu):
            cont = pl.contourf(xg, yg, cols,ncont,corner_mask = True,cmap=cmap)
        else:
            levels = np.linspace(minu,maxu,ncont+1)
            cols[(cols > maxu)] = maxu
            cols[(cols < minu)] = minu
            cont = pl.contourf(xg, yg, cols,levels=levels,corner_mask = True,cmap=cmap)
        #cs = pl.contour(xg, yg, cols,ncont,cmap=cmap)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)
    else:
        ext=(xgri[0,0]-steplo/2, xgri[-1,0]+steplo/2, ygri[0,0]-steplo/2, ygri[0,-1]+steplo/2)
        if(minu == maxu):
            if npoints:
                lvls = np.logspace(0,3,10)
                if not image: cont = pl.contourf(xgri, ygri, num,norm=LogNorm(),levels=lvls,corner_mask = True,cmap=cmap)
                if image: pl.imshow(num, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest',cmap=cmap)
            else:
                if not image: cont = pl.contourf(xgri, ygri, cols,ncont,corner_mask = True,cmap=cmap)
                if image: pl.imshow(cols, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest',cmap=cmap)#, cmap=cm.gist_rainbow)
        else:
            cols[(cols > maxu)] = maxu
            cols[(cols < minu)] = minu
            levels = np.linspace(minu,maxu,ncont+1)
            if normalize:
                levels = levels/np.max(levels)
                cols = cols/np.max(levels)
                cbarmult = None
                if print_values:
                    filosname = nomefi[:nomefi.index('.')]+'.dat'
                    filos = open(filosname,'w')
                    filos.write('#{:7s}{:8s}{:12s}\n'.format('Lat','Lon','Value'))
                    for xxi,yyi,co in zip(xgri.flatten(),ygri.flatten(),cols.flatten()):
                        lo, la = map(xxi, yyi, inverse=True)
                        if abs(la) < np.min(abs(blats)): continue
                        try:
                            filos.write('{:8.2f}  {:8.2f}  {:12.3f}\n'.format(la,lo,co))
                        except Exception as cazzillo:
                            print(cazzillo)
                            print(type(la),type(lo),type(co))
                            print('PROBLEMA, converto masked a nan: {} {} {}'.format(la,lo,co))
                            filos.write('{:8.2f}  {:8.2f}  {:12.3f}\n'.format(la,lo,np.nan))
                    filos.close()
            if npoints:
                lvls = np.logspace(0,3,10)
                if not image: cont = pl.contourf(xgri, ygri, num,norm=LogNorm(),levels=lvls,corner_mask = True,cmap=cmap)
                if image: pl.imshow(num, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest')#, cmap=cm.gist_rainbow)
            else:
                print(ext)
                #pl.scatter(xgri[0,0],ygri[0,0],s=10)
                if image: pl.imshow(cols.T, origin='lower', extent=ext,interpolation='nearest')#, cmap=cm.gist_rainbow)
                if not image: cont = pl.contourf(xgri, ygri, cols,levels=levels,corner_mask = True,cmap=cmap)
        #cs = pl.contour(xgri, ygri, cols,ncont,cmap=cmap)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)

    if cbarmult is not None:
        cbarlabel = cbarlabel.format(cbarmult)

    if cbarmult is not None and cbarform is not None:
        try:
            ticks = levels[::salta]
        except:
            levels = cont.levels
            ticks = levels[::salta]

        ticklab = [('{:'+cbarform[1:]+'}').format(ti/10**cbarmult) for ti in ticks]
        cb = pl.colorbar(ax=ax, pad = 0.1, shrink = 0.8, aspect = 15)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklab)
        cb.set_label(cbarlabel,fontsize=style)
    else:
        if cbarform is not None:
            cb = pl.colorbar(ax=ax,format=cbarform, pad = 0.1, shrink = 0.8, aspect = 15)
            cb.set_label(cbarlabel,fontsize=style)
        else:
            cb = pl.colorbar(ax=ax,pad = 0.1, shrink = 0.8, aspect = 15)
            cb.set_label(cbarlabel,fontsize=style)

    cb.ax.tick_params(labelsize=fnsz)

    if addpoints:
        if(condpo is None):
            if(minu == maxu):
                print(np.min(cols[~np.isnan(cols)]),np.max(cols[~np.isnan(cols)]))
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=np.min(cols[~np.isnan(cols)]),vmax=np.max(cols[~np.isnan(cols)]),cmap=cmap)
            else:
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=minu,vmax=maxu,cmap=cmap)
        else:
            print('cisiamo')
            print(len(x[condpo]))
            print(x)
            print(condpo)
            print(len(condpo))
            sca = pl.scatter(x[condpo],y[condpo],color='black',marker='o',s=1)

    #pl.scatter(xgri[conan],ygri[conan],color='black',marker='o',s=1)

    vip4x, vip4y = map(360-vip4_lon,vip4_lat)

    vip4x = np.append(vip4x,vip4x[0])
    vip4y = np.append(vip4y,vip4y[0])
    stat_x = np.append(stat_x,stat_x[0])
    stat_y = np.append(stat_y,stat_y[0])
    aur_lw=2.0
    if aur_model == 'VIP4':
        pl.plot(vip4x,vip4y,color='white',linewidth=aur_lw)
        pl.plot(vip4x,vip4y,color='black',linewidth=aur_lw,linestyle='--')
    elif aur_model == 'stat':
        pl.plot(stat_x,stat_y,color='white',linewidth=aur_lw)
        pl.plot(stat_x,stat_y,color='black',linewidth=aur_lw,linestyle='--')
    else:
        pl.plot(vip4x,vip4y,color='white',linewidth=aur_lw)
        pl.plot(vip4x,vip4y,color='black',linewidth=aur_lw,linestyle='--')
        pl.plot(stat_x,stat_y,color='white',linewidth=aur_lw)
        pl.plot(stat_x,stat_y,color='black',linewidth=aur_lw,linestyle='-')

    if live: show=True
    if(show): pl.show()
    if not live and axu is None:
        if nomefi is not None:
            lol = nomefi.find('.')
            form = nomefi[lol+1:]
            fig.savefig(nomefi, format=form, dpi=150)
        if pdf is not None: pdf.savefig(fig)
        pl.close()

    return


def stereopos(lon,lat,nomefi,color='black',marker='.',polo='N',title='',show=False,proj='orto'):
    """
    Plots points on a stereographic map with colors.
    :return:
    """

    rsp = [71492.,66854.]
    fig = pl.figure(figsize=(8, 6), dpi=150)
    pl.title(title)

    if polo == 'N':
        blat = 55
        xmi = rsp[0]*np.cos(np.pi*blat/180.0)
        if proj == 'orto': map = Basemap(projection='ortho',lon_0=0,lat_0=90,resolution='l',rsphere=rsp[0],
                                        llcrnrx=-xmi,llcrnry=-xmi,urcrnrx=xmi,urcrnry=xmi)
        if proj == 'stereo': map = Basemap(projection='npstere',boundinglat=blat,lon_0=0,lat_0=90,resolution='l')
        map.drawparallels(np.arange(0,90,10))
    else:
        blat = -60
        xmi = rsp[0]*np.cos(np.pi*blat/180.0)
        if proj == 'orto': map = Basemap(projection='ortho',boundinglat=55,lon_0=180,lat_0=-90,resolution='l',rsphere=rsp[0],
                                llcrnrx=-xmi,llcrnry=-xmi,urcrnrx=xmi,urcrnry=xmi)
        if proj == 'stereo': map = Basemap(projection='spstere',boundinglat=blat,lon_0=180,lat_0=-90,resolution='l')
        map.drawparallels(np.arange(-80,0,10))

    aur_lon_0,aur_lat,aur_lon,aur_theta = leggi_map_aur(polo)

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,1,1],fontsize=10,fmt=fmt)

    x, y = map(lon,lat)

    sca = map.scatter(x,y,s=1,marker=marker,color=color)
    # x = np.append(x,x[0])
    # y = np.append(y,y[0])
    # pl.plot(x,y)

    x, y = map(360-aur_lon,aur_lat)
    #map.scatter(x,y,color = 'white',edgecolors='black',s=15,marker = 'o')
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    pl.plot(x,y,color='white',linewidth=2.0)
    pl.plot(x,y,color='black',linewidth=2.0,linestyle='--')
    if(show): pl.show()
    fig.savefig(nomefi, format='eps', dpi=150)
    pl.close()
    return


def findspi(wls,spe,thres=1.5,min=-2e-4):
    """
    Sets the mask at 0 if the point is a suspected spike. Checks line intensities on the H3+ line and each other point with more than thres the max of these lines is masked.
    :return: mask
    """

    lines = np.array([3315.0,3413,3531,3540,3665])
    vals = np.zeros(5)
    for line,i in zip(lines,range(5)):
        cond1 = (abs(wls-line) < 1.0)
        vals[i] = spe[cond1]

    cond = ((wls > 3200.) & (wls < 3800.)) & ((spe > thres*np.max(vals)) | (spe < min))
    mask = np.ones(len(spe), dtype = 'i4')
    mask[cond] = 0

    return mask


def checkqual(wls,spe,fondo,thres=-1e-3):
    """
    Checks the quality of the spectrum, counting elements lower than threshold with respect to the continuum.
    :return: 1 if ok, 0 in not
    """

    cond = (wls > 3250.) & (wls < 3700.) & (spe - fondo < thres)
    nu = sum(cond)

    mask = findspi(wls,spe)
    spi = len(mask[(mask == 0)])

    if(nu > 2 or spi > 2):
        ok = 0
    else:
        ok = 1

    return ok


def stereopolar(lon,lat,R=71000):
    """
    Calculates the stereographic projection, given lon, lat and radius.
    :return:
    """
    lamb = np.pi*0.5
    phi = np.pi
    lat = np.pi*lat/180.0
    lon = np.pi*lon/180.0

    k = 2*R/(1+m.sin(phi)*m.sin(lon)+m.sin(phi)*m.cos(lon)*m.cos(lat-lamb))
    x = k * m.cos(phi)*m.sin(lat-lamb)
    y = k * (m.cos(phi)*m.sin(lon)-m.sin(phi)*m.cos(lon)*m.cos(lat-lamb))

    return x,y


def scatter(x,y,col,cond=None,lat=None,lon=None,polo='S',xlim=None,ylim=None,clim=None,nome=None,
                xlabel='',ylabel='',title='',live=False,pdf=None):
    """
    Scatter plot con condizione. Mostra lo scatter plot, evidenziando i punti considerati e stampa la mappa corrispondente.
    :return:
    """
    fig = pl.figure(figsize=(8, 6), dpi=150)
    if xlim is not None: pl.xlim(xlim[0],xlim[1])
    if ylim is not None: pl.ylim(ylim[0],ylim[1])
    vmin = None
    vmax = None
    if clim is not None:
        vmin = clim[0]
        vmax = clim[1]

    pl.title(title)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.grid()

    if cond is not None:
        pl.scatter(x[~cond],y[~cond],color='grey',s=2,edgecolor='None',vmin=vmin,vmax=vmax)
        pl.scatter(x[cond],y[cond],c=col[cond],s=4,edgecolor='None',vmin=vmin,vmax=vmax)
    else:
        pl.scatter(x,y,c=col,s=4,edgecolor='None',vmin=vmin,vmax=vmax)

    pl.colorbar()

    if live: pl.show()
    if nome is not None: fig.savefig(nome, format='eps', dpi=150)
    if pdf is not None: pdf.savefig(fig)
    #pl.close()


    #fig2 = pl.figure(figsize=(8, 6), dpi=150)
    if lon is not None and cond is not None:
        if nome is not None:
            nomefi = nome[0:nome.find('.eps')]+'_MAP.eps'
            stereoplot(lon,lat,col,nomefi=nomefi,polo=polo,minu=vmin,maxu=vmax,condpo=cond,live=live,pdf=pdf)
        else:
            stereoplot(lon,lat,col,polo=polo,minu=vmin,maxu=vmax,condpo=cond,live=live,pdf=pdf)

    if not live: pl.close()

    return

def leggi_map_aur(polo):
    """
    VIP4:
    Legge le coordinate dell'ovale previsto dal modello VIP4.
    (Connerney, J.E.P., Açuna, M.H., Ness N.F., & Satoh, T. (1998).
    New models of Jupiter's magnetic field constrained by the Io Flux Tube footprint. J. Geophys. Res., 103, 11929 -11939)
    stat:
    Legge le coordinate dell'ovale statistico.
    :param polo:
    :return:
    """
    if polo == 'S':
        filename = '/home/fedefab/Scrivania/Research/Jiram/DATA/Model_VIP4/southR30table.txt'
    if polo == 'N':
        filename = '/home/fedefab/Scrivania/Research/Jiram/DATA/Model_VIP4/northR30table.txt'

    infile = open(filename, 'r')
    infile.readline()
    infile.readline()
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    aur_lon_0 = np.array([float(r) for r in data_arr[:, 0]])
    aur_lat = np.array([float(r) for r in data_arr[:, 1]])
    aur_lon = np.array([float(r) for r in data_arr[:, 2]])
    aur_theta = np.array([float(r) for r in data_arr[:, 3]])
    infile.close()

    if polo == 'S':
        filename = '/home/fedefab/Scrivania/Research/Jiram/DATA/Model_VIP4/statistico/south_aurora.txt'
    if polo == 'N':
        filename = '/home/fedefab/Scrivania/Research/Jiram/DATA/Model_VIP4/statistico/north_aurora.txt'

    infile = open(filename, 'r')
    infile.readline()
    infile.readline()
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    aur_x = np.array([float(r) for r in data_arr[:, 0]])
    aur_y = np.array([float(r) for r in data_arr[:, 1]])
    infile.close()
    rsp = [71492.,66854.]

    if polo == 'N':
        aur_x = - aur_x * rsp[0]
        aur_y = - aur_y * rsp[0] # il segno di y è negativo verso i 180 nel plot di ale
    else:
        aur_x = aur_x * rsp[0]
        aur_y = aur_y * rsp[0]

    return aur_x,aur_y,aur_lat,aur_lon


def fondojir(wl,spe):
    """
    Calculates the mean value of the background radiation in the spectrum.
    :param wl: Wavelengths
    :param spe: Spectrum
    :return:
    """
    cond = ((wl > 3475) & (wl < 3511)) | ((wl > 3558) & (wl < 3603)) | ((wl > 3735) & (wl < 3770))

    fondo = np.mean(spe[cond])

    return fondo


def ratio32_35(wl,spe,fondo):
    """
    Calculates ratio between H3+ lines.
    :param wl:
    :param spe:
    :return:
    """
    thres = 6e-4

    #cond1 = ((wl > 3195) & (wl < 3201))
    #cond2 = ((wl > 3518) & (wl < 3546))

    int1 = spe[133]-np.mean(spe[131:133])
    int1 = spe[133]+spe[134]-2*np.mean(spe[131:133])
    int2 = spe[171]+spe[170]-2*np.mean(spe[166:169])
    #int1 = spe[171] - fondo*2e3+(spe[170])
    int3 = spe[185]+spe[186]-2*np.mean(spe[182:185])
    ratio = int1/int2
    err = 3e-4/int1+3e-4/int2
    err = ratio*err
    if(int1 < thres or int2 < thres):
        ratio = float(np.nan)
        err = float(np.nan)

#    print(int1,int2,ratio,err)

    return ratio, err


def ratio35_37(wl,spe,fondo):
    """
    Calculates ratio between H3+ lines.
    :param wl:
    :param spe:
    :return:
    """
    thres = 6e-4

    #cond1 = ((wl > 3195) & (wl < 3201))
    #cond2 = ((wl > 3518) & (wl < 3546))

    int1 = spe[133]-np.mean(spe[131:133])
    int1 = spe[133]+spe[134]-2*np.mean(spe[131:133])
    int2 = spe[171]+spe[170]-2*np.mean(spe[166:169])
    #int1 = spe[171] - fondo*2e3+(spe[170])
    int3 = spe[185]+spe[186]-2*np.mean(spe[182:185])
    ratio = int2/int3
    err = 3e-4/int2+3e-4/int3
    err = ratio*err
    if(int2 < thres or int3 < thres):
        ratio = float(np.nan)
        err = float(np.nan)

#    print(int1,int2,ratio,err)

    return ratio, err


def ratio32_37(wl,spe,fondo):
    """
    Calculates ratio between H3+ lines.
    :param wl:
    :param spe:
    :return:
    """
    thres = 6e-4

    #cond1 = ((wl > 3195) & (wl < 3201))
    #cond2 = ((wl > 3518) & (wl < 3546))

    int1 = spe[133]-np.mean(spe[131:133])
    int1 = spe[133]+spe[134]-2*np.mean(spe[130:133])
    int2 = spe[171]+spe[170]-2*np.mean(spe[166:169])
    #int1 = spe[171] - fondo*2e3+(spe[170])
    int3 = spe[185]+spe[186]-2*np.mean(spe[182:185])
    ratio = int1/int3
    err = 3e-4/int2+3e-4/int3
    err = ratio*err
    if(int1 < thres or int3 < thres):
        ratio = float(np.nan)
        err = float(np.nan)

    return ratio, err


def ind_h3p(wl,spe,fondo):
    """
    Integrates the signal from H3+ lines.
    :param wl:
    :param spe:
    :return:
    """

    cond = ((wl > 3250) & (wl < 3275)) | ((wl > 3293) & (wl < 3330)) | ((wl > 3375) & (wl < 3463)) | ((wl > 3518) & (wl < 3546)) | ((wl > 3607) & (wl < 3630)) | ((wl > 3653) & (wl < 3681))

    intt = np.sum(spe[cond]-fondo)

    return intt


def integr_h3p(wl,spe,fondo,w1=3350,w2=3750):
    """
    Integrates the signal from H3+ lines.
    :param wl:
    :param spe:
    :return:
    """

    cond = (wl > w1) & (wl < w2)

    intt = np.trapz(spe[cond]*1e-3-fondo,x=wl[cond])

    return intt


def write_obs_JIR(freq,spe,mask,filename,comment=''):
    """
    Writes files of JIRAM observations. (JIRAM_MAP format)
    :return:
    """
    from datetime import datetime
    infile = open(filename, 'w')
    data = datetime.now()
    infile.write(comment+'\n')
    infile.write('\n')
    infile.write('Processed on: {}\n'.format(data))
    infile.write('\n')
    infile.write('Wavelength (nm), spectral data (W m^-2 um^-1 sr^-1), mask(0/1):\n')
    infile.write('{:1s}\n'.format('#'))

    for fr, ob, ma in zip(freq, spe, mask):
        if(np.isnan(ob)): ob = 0.0
        if(ma == 0): ob = 0.0
        infile.write('{:10.3f}{:15.5e}{:4d}\n'.format(fr,ob,ma))

    infile.close()
    return


def read_sim_jir(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    wl = np.array([float(r) for r in data_arr[:, 0]])
    data = np.array([float(r) for r in data_arr[:, 1]])
    sim = np.array([float(r) for r in data_arr[:, 2]])
    infile.close()
    return wl, data, sim


def read_res_jir(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    i = np.array([int(r) for r in data_arr[:, 0]])
    j = np.array([int(r) for r in data_arr[:, 1]])
    temp = np.array([float(r) for r in data_arr[:, 2]])
    err_t = np.array([float(r) for r in data_arr[:, 3]])
    infile.close()
    return i,j,temp,err_t


def read_res_jir_2(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    i = np.array([int(r) for r in data_arr[:, 0]])
    j = np.array([int(r) for r in data_arr[:, 1]])
    temp = np.array([float(r) for r in data_arr[:, 2]])
    infile.close()
    return i,j,temp

def read_res_jir_3(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    i = np.array([int(r) for r in data_arr[:, 0]])
    temp = np.array([float(r) for r in data_arr[:, 1]])
    err_t = np.array([float(r) for r in data_arr[:, 2]])
    infile.close()
    return i,temp,err_t


def read_res_jir_4(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    i = np.array([int(r) for r in data_arr[:, 0]])
    temp = np.array([float(r) for r in data_arr[:, 1]])
    infile.close()
    return i,temp


def leggi_console(polo,cartres=None,filtra=True):

    if polo == 'S':
        input_file = '/home/fedefab/Scrivania/Research/Dotto/Git/JiramPy/input_mappe_S.in'
    elif polo == 'N':
        input_file = '/home/fedefab/Scrivania/Research/Dotto/Git/JiramPy/input_mappe_N.in'
    else:
        print('puppa')
        sys.exit()

    ######################################################################################à

    # try:
    #     input_file = sys.argv[1]
    # except:
    #     raise ValueError('You have to insert the input file path to run the program, in this way: ----->     python mappe.py /path/to/input_file   <------')

    keys = 'cart cart80 cartres cartout picfile picfile500 picfile80 polo lch4 l500 lN80 lshi maxchi mcol mincol tmax npix aurm emissMAX cstep tstep errmax maxint maxch4 stepint stepch4'
    keys = keys.split()
    types = [str,str,str,str,str,str,str,str,bool,bool,bool,bool,float,float,float,float,int,str,float,float,float,float,float,float,float,float]

    values = sbm.read_inputs(input_file,keys,itype=types,verbose=True)

    cart = values['cart']
    cart80 = values['cart80']
    cartres = values['cartres']
    picfile500 = values['picfile500']
    picfile = values['picfile']
    picfile80 = values['picfile80']
    cartout = values['cartout']
    if not os.path.exists(cartout): os.makedirs(cartout)

    polo = values['polo']
    lch4 = values['lch4']
    l500 = values['l500']
    lN80 = values['lN80']
    lshi = values['lshi']

    maxchi = values['maxchi']
    mcol = values['mcol']
    mincol = values['mincol']
    tmax = values['tmax']
    npix = values['npix']
    aurm = values['aurm']
    emissMAX = values['emissMAX']
    cstep = values['cstep']
    tstep = values['tstep']
    errmax = values['errmax']
    maxint = values['maxint']
    maxch4 = values['maxch4']
    stepint = values['stepint']
    stepch4 = values['stepch4']

    cbarlabcol = r'$H_3^+$ column ($\times 10^{{{}}}$ cm$^{{-2}}$)'
    cbarlaberrcol = 'Relative error'
    cbarlabtemp = r'$H_3^+$ temperature (K)'
    cbarlaberrtemp = r'Error on $H_3^+$ temperature (K)'
    cbarlabch4 = r'CH$_4$ column ($\times 10^{{{}}}$ cm$^{{-2}}$)'

    cbarlabint = r'Integrated intensity ($\times 10^{{{}}}$ $W {{m}}^{{-2}} {{sr}}^{{-1}}$)'
    titleintegr = 'Integrated intensity'

    titlecol = r'$H_3^+$ column'
    titleerrcol = r'Relative error on $H_3^+$ column'
    titletemp = r'$H_3^+$ temperature'
    titleerrtemp = r'Error on $H_3^+$ temperature'
    titlech4 = r'CH$_4$ column'

    ####################################################################################
    ####################                                   #############################
    #############                   LEGGE                         #######################
    ###### LEGGE ########################################################################

    pixs = pickle.load(open(picfile,'r'))

    if l500:
        # si va a prendere le geometrie per 500 km
        p500 = pickle.load(open(picfile500,'r'))
        pixs500 = p500.view(np.recarray)
        print(len(pixs),len(pixs500))
        #print(pixs[1139])
        #print(pixs500[1139])

    ii,col1, err_col = read_res_jir_3(cartres+'CD-H3p.dat')
    iit,temp1, err_temp = read_res_jir_3(cartres+'VT-H3p.dat')
    iic,chi1 = read_res_jir_4(cartres+'chisq.dat')
    iio,off1 = read_res_jir_4(cartres+'offset_ok.dat')
    if lshi: iis,shi1 = read_res_jir_4(cartres+'shift_ok.dat')
    if lch4: ii4,ch41, err_ch41 = read_res_jir_3(cartres+'CD-CH4.dat')

    col = np.zeros(len(pixs))
    err_c = np.zeros(len(pixs))
    temp = np.zeros(len(pixs))
    err_t = np.zeros(len(pixs))
    chi = np.zeros(len(pixs))
    off = np.zeros(len(pixs))
    shi = np.zeros(len(pixs))
    ch4 = np.zeros(len(pixs))
    err_ch4 = np.zeros(len(pixs))

    col[ii] = col1
    err_c[ii] = err_col
    temp[iit] = temp1
    err_t[iit] = err_temp
    chi[iic] = chi1
    chi[chi == 0.0] = np.nan
    off[iio] = off1
    if lshi: shi[iis] = shi1
    if lch4:
        ch4[ii4] = ch41
        err_ch4[ii4] = err_ch41

    if lN80:
        p80 = pickle.load(open(picfile80,'r'))
        ii,col1, err_col = read_res_jir_3(cart80+'CD-H3p.dat')
        iit,temp1, err_temp = read_res_jir_3(cart80+'VT-H3p.dat')
        iic,chi1 = read_res_jir_4(cart80+'chisq.dat')
        iio,off1 = read_res_jir_4(cart80+'offset_ok.dat')
        if lshi: iis,shi1 = read_res_jir_4(cart80+'shift_ok.dat')
        if lch4: ii4,ch41, err_ch41 = read_res_jir_3(cart80+'CD-CH4.dat')

        col80 = np.zeros(len(p80))
        err_c80 = np.zeros(len(p80))
        temp80 = np.zeros(len(p80))
        err_t80 = np.zeros(len(p80))
        chi80 = np.zeros(len(p80))
        off80 = np.zeros(len(p80))
        shi80 = np.zeros(len(p80))
        ch480 = np.zeros(len(p80))
        err_ch480 = np.zeros(len(p80))

        col80[ii] = col1
        err_c80[ii] = err_col
        temp80[iit] = temp1
        err_t80[iit] = err_temp
        chi80[iic] = chi1
        chi80[chi80 == 0.0] = np.nan
        off80[iio] = off1
        if lshi: shi80[iis] = shi1
        if lch4:
            ch480[ii4] = ch41
            err_ch480[ii4] = err_ch41

        #print(type(pixs),len(pixs))
        p80 = p80.view(np.recarray)
        co80 = (p80.emiss_angle < emissMAX)
        p80 = p80[co80]
        pixs = np.append(pixs,p80)
        pixs500 = np.append(pixs500,p80)
        #print(type(pixs),len(pixs))
        pixs = pixs.view(np.recarray)
        pixs500 = pixs500.view(np.recarray)
        #print(type(pixs),len(pixs))
        col = np.append(col,col80[co80])
        err_c = np.append(err_c,err_c80[co80])
        temp = np.append(temp,temp80[co80])
        err_t = np.append(err_t,err_t80[co80])
        chi = np.append(chi,chi80[co80])
        off = np.append(off,off80[co80])
        shi = np.append(shi,shi80[co80])
        ch4 = np.append(ch4,ch480[co80])
        err_ch4 = np.append(err_ch4,err_ch480[co80])

    #print(type(pixs.cubo))
    cubs = np.unique(pixs500.cubo)
    i=0
    for cu in cubs:
        if not np.all(np.isnan(pixs500[pixs500.cubo == cu].pc_lon)):
            em = np.mean(pixs500[pixs500.cubo == cu].emiss_angle)
            so = np.mean(pixs500[pixs500.cubo == cu].solar_time)
            n = len(pixs500[pixs500.cubo == cu])
            print('{}  {}  {:5.1f}  {:5.1f} {}'.format(i,cu,em,so,n))

    ######## fine letture ################################################################################

    # Faccio le mappe: #######################################

    cond2 = ((chi > maxchi) | (col <= 0.0) | (off < 0.0))

    col[cond2] = float('NaN')
    print(len(col[cond2]))
    temp[cond2] = float('NaN')
    chi[cond2] = float('NaN')
    off[cond2] = float('NaN')
    err_t[cond2] = float('NaN')
    err_c[cond2] = float('NaN')

    if lch4: ch4[cond2] = float('NaN')
    if lch4: ch4[(ch4<0.0)] = 0.0

    if l500: pixs = pixs500
    indx = np.arange(len(pixs))

    if filtra:
        nonan = (~np.isnan(col)) & (~np.isnan(pixs.pc_lon))
        col = col[nonan]
        temp = temp[nonan]
        chi = chi[nonan]
        off = off[nonan]
        err_t = err_t[nonan]
        err_c = err_c[nonan]
        ch4 = ch4[nonan]
        err_ch4 = err_ch4[nonan]
        shi = shi[nonan]
        pixs = pixs[nonan]
        indx = indx[nonan]

    col_c = col*np.cos(np.pi*pixs.emiss_angle/180.0)

    return pixs,col,col_c,err_c,temp,err_t,chi,off,shi,ch4,err_ch4,nonan


def scatt_color(x,y,file=None,c=None,xlabel='',ylabel='',title=''):
    """
    Makes a scatter plot with color scale.
    :return:
    """
    #
    return
