#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import matplotlib.lines as lin
import math as m
from numpy import linalg as LA
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import pickle


def stereoplot(lon,lat,col,nomefi=None,live = False,polo='N',min=None,max=None,title='',show=False,aur_model='stat',
               proj='orto',condpo=None, pdf=None):
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
        if proj == 'orto': map = Basemap(projection='ortho',boundinglat=55,lon_0=0,lat_0=-90,resolution='l',rsphere=rsp[0],
                                llcrnrx=-xmi,llcrnry=-xmi,urcrnrx=xmi,urcrnry=xmi)
        if proj == 'stereo': map = Basemap(projection='spstere',boundinglat=blat,lon_0=180,lat_0=-90,resolution='l')
        map.drawparallels(np.arange(-80,0,10))

    stat_x,stat_y,vip4_lat,vip4_lon = leggi_map_aur(polo)
    if polo == 'N':
        x0, y0 = map(0.,90.)
    else:
        x0, y0 = map(0.,-90.)
    stat_x = stat_x+x0
    stat_y = stat_y+y0

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,0,1],fontsize=10)

    x, y = map(lon,lat)

    if condpo is None:
        sca = map.scatter(x,y,c = col,s=5,edgecolors='none',vmin=min,vmax=max)
    else:
        sca1 = map.scatter(x[~condpo],y[~condpo],s=2,color='grey',edgecolors='none')
        sca2 = map.scatter(x[condpo],y[condpo],c=col[condpo],s=5,edgecolors='none',vmin=min,vmax=max)

    pl.colorbar()

    vip4x, vip4y = map(360-vip4_lon,vip4_lat)

    vip4x = np.append(vip4x,vip4x[0])
    vip4y = np.append(vip4y,vip4y[0])
    stat_x = np.append(stat_x,stat_x[0])
    stat_y = np.append(stat_y,stat_y[0])
    if aur_model == 'VIP4':
        pl.plot(vip4x,vip4y,color='white',linewidth=2.0)
        pl.plot(vip4x,vip4y,color='black',linewidth=2.0,linestyle='--')
    elif aur_model == 'stat':
        pl.plot(stat_x,stat_y,color='white',linewidth=2.0)
        pl.plot(stat_x,stat_y,color='black',linewidth=2.0,linestyle='--')
    else:
        pl.plot(vip4x,vip4y,color='white',linewidth=2.0)
        pl.plot(vip4x,vip4y,color='black',linewidth=2.0,linestyle=':')
        pl.plot(stat_x,stat_y,color='white',linewidth=2.0)
        pl.plot(stat_x,stat_y,color='black',linewidth=2.0,linestyle='--')

    if live: show=True
    if(show): pl.show()
    if not live:
        if nomefi is not None: fig.savefig(nomefi, format='eps', dpi=150)
        if pdf is not None: pdf.savefig(fig)
        pl.close()

    return


def stereomap(lon,lat,col,nomefi=None,polo='N',min=0,max=0,title='',show=False,lonlat=False,xres=50,lonres=180,latres=30,
              ncont=15,form=False,addpoints=False,condpo=None,divide=False):
    """
    Plots points on a stereographic map with colors.
    :return:
    """
    fig = pl.figure(figsize=(8, 6), dpi=150)
    pl.title(title)

    if polo == 'N':
        blat = 60
        map = Basemap(projection='npstere',boundinglat=blat,lon_0=180,resolution='l')
        map.drawparallels(np.arange(blat,90,10))
    else:
        blat = -60
        map = Basemap(projection='spstere',boundinglat=blat,lon_0=180,resolution='l')
        map.drawparallels(np.arange(-80,blat+10,10))

    aur_lon_0,aur_lat,aur_lon,aur_theta = leggi_map_aur(polo)

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,0,1],fontsize=10)

    x, y = map(lon,lat)
    # print(x,y)
    # print(map(0.,-60.),map(90.,-60.),map(180.,-60.),map(270.,-60.))

    # Trovo i minimi e massimi delle coordinate del plot
    xu=np.zeros(4)
    yu=np.zeros(4)
    xu[0],yu[0] = map(0.,blat)
    xu[1],yu[1] = map(90.,blat)
    xu[2],yu[2] = map(180.,blat)
    xu[3],yu[3] = map(270.,blat)
    x0=np.min(xu)
    x1=np.max(xu)
    y0=np.min(yu)
    y1=np.max(yu)

    nx=xres
    ny=xres
    nstep=xres*1j
    xgri, ygri = np.mgrid[x0:x1:nstep, y0:y1:nstep]

    nsteplot=lonres*1j
    nsteplat=latres*1j
    if polo == 'N':
        grid_lat, grid_lon = np.mgrid[blat:90:nsteplat, 0:360:nsteplot]
    else:
        grid_lat, grid_lon = np.mgrid[-90:blat:nsteplat, 0:360:nsteplot]
    xg, yg = map(grid_lon, grid_lat)

    nlat = latres #np.shape(grid_lat)[0]
    nlon = lonres #np.shape(grid_lat)[1]
    steplo = 360/nlon
    stepla = 30/nlat

    # tolgo i NaN dai vettori
    cond2 = (~np.isnan(col))
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
                if len(col2[cond]) > 0:
                    cols[i,j] = np.mean(col2[cond])
                num[i,j] = len(col2[cond])

    cols[(cols<0)]=float(np.nan)

    #pl.pcolormesh(xg, yg, cols)

    if lonlat:
        if(min == max):
            pl.contourf(xg, yg, cols,ncont)
        else:
            levels = np.linspace(min,max,ncont+1)
            cols[(cols > max)] = max
            cols[(cols < min)] = min
            pl.contourf(xg, yg, cols,levels=levels)
        #cs = pl.contour(xg, yg, cols,ncont)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)
    else:
        if(min == max):
            pl.contourf(xgri, ygri, cols,ncont)
        else:
            cols[(cols > max)] = max
            cols[(cols < min)] = min
            levels = np.linspace(min,max,ncont+1)
            if divide:
                pl.contourf(xgri, ygri, cols/np.sqrt(num),levels=levels)
            else:
                pl.contourf(xgri, ygri, cols,levels=levels)
        #cs = pl.contour(xgri, ygri, cols,ncont)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)

    if form:
        pl.colorbar(format='%.1e')
    else:
        pl.colorbar()

    if addpoints:
        if(condpo is None):
            if(min == max):
                print(np.min(cols[~np.isnan(cols)]),np.max(cols[~np.isnan(cols)]))
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=np.min(cols[~np.isnan(cols)]),vmax=np.max(cols[~np.isnan(cols)]))
            else:
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=min,vmax=max)
        else:
            print('cisiamo')
            print(len(x[condpo]))
            print(x)
            print(condpo)
            print(len(condpo))
            sca = pl.scatter(x[condpo],y[condpo],color='black',marker='o',s=1)


    #pl.colorbar()
    x, y = map(360-aur_lon,aur_lat)
    #map.scatter(x,y,color = 'white',edgecolors='black',s=15,marker = 'o')
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    pl.plot(x,y,color='white',linewidth=2.0)
    pl.plot(x,y,color='black',linewidth=2.0,linestyle='--')
    if(show): pl.show()
    if nomefi is not None: fig.savefig(nomefi, format='eps', dpi=150)
    pl.close()

    return


def stereomap2(lon,lat,col,nomefi=None,polo='N',proj = 'orto',min=0,max=0,title='',show=False,lonlat=False,xres=50,lonres=180,
               latres=30,ncont=15,form=False,addpoints=False,condpo=None,minnum=2,image=False,
               interp='nearest',npoints=False,aur_model='stat',live=False, pdf=None):
    """
    Plots points on an ortho- or a stereographic map with colors.
    :return:
    """
    from matplotlib.colors import LogNorm

    rsp = [71492.,66854.]
    pl.close()
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
        if proj == 'orto': map = Basemap(projection='ortho',boundinglat=55,lon_0=0,lat_0=-90,resolution='l',rsphere=rsp[0],
                                llcrnrx=-xmi,llcrnry=-xmi,urcrnrx=xmi,urcrnry=xmi)
        if proj == 'stereo': map = Basemap(projection='spstere',boundinglat=blat,lon_0=180,lat_0=-90,resolution='l')
        map.drawparallels(np.arange(-80,0,10))

    stat_x,stat_y,vip4_lat,vip4_lon = leggi_map_aur(polo)
    if polo == 'N':
        x0, y0 = map(0.,90.)
    else:
        x0, y0 = map(0.,-90.)
    stat_x = stat_x+x0
    stat_y = stat_y+y0

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,0,1],fontsize=10)

    x, y = map(lon,lat)
    # print(x,y)
    # print(map(0.,-60.),map(90.,-60.),map(180.,-60.),map(270.,-60.))

    # Trovo i minimi e massimi delle coordinate del plot
    xu=np.zeros(4)
    yu=np.zeros(4)
    xu[0],yu[0] = map(0.,blat)
    xu[1],yu[1] = map(90.,blat)
    xu[2],yu[2] = map(180.,blat)
    xu[3],yu[3] = map(270.,blat)
    x0=np.min(xu)
    x1=np.max(xu)
    y0=np.min(yu)
    y1=np.max(yu)

    nx=xres
    ny=xres
    nstep=xres*1j
    xgri, ygri = np.mgrid[x0:x1:nstep, y0:y1:nstep]

    nsteplot=lonres*1j
    nsteplat=latres*1j
    if polo == 'N':
        grid_lat, grid_lon = np.mgrid[blat:90:nsteplat, 0:360:nsteplot]
    else:
        grid_lat, grid_lon = np.mgrid[-90:blat:nsteplat, 0:360:nsteplot]
    xg, yg = map(grid_lon, grid_lat)

    nlat = latres #np.shape(grid_lat)[0]
    nlon = lonres #np.shape(grid_lat)[1]
    steplo = 360/nlon
    stepla = 30/nlat

    # tolgo i NaN dai vettori
    cond2 = (~np.isnan(col))
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
                if len(col2[cond]) > 0:
                    cols[i,j] = np.mean(col2[cond])
                num[i,j] = len(col2[cond])

    cols[(cols<0)]=float(np.nan)
    cols[(num < minnum)]=float(np.nan)
    #cols = np.ma.masked_array(cols, mask=cols<0.0)

    #pl.pcolormesh(xg, yg, cols)

    if lonlat:
        if(min == max):
            pl.contourf(xg, yg, cols,ncont)
        else:
            levels = np.linspace(min,max,ncont+1)
            cols[(cols > max)] = max
            cols[(cols < min)] = min
            pl.contourf(xg, yg, cols,levels=levels)
        #cs = pl.contour(xg, yg, cols,ncont)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)
    else:
        ext=(xgri[0,0]-steplo/2, xgri[-1,0]+steplo/2, ygri[0,0]-steplo/2, ygri[0,-1]+steplo/2)
        if(min == max):
            if npoints:
                lvls = np.logspace(0,3,10)
                if not image: pl.contourf(xgri, ygri, num,norm=LogNorm(),levels=lvls)
                if image: pl.imshow(num, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest')
            else:
                if not image: pl.contourf(xgri, ygri, cols)
                if image: pl.imshow(cols, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest')#, cmap=cm.gist_rainbow)
        else:
            cols[(cols > max)] = max
            cols[(cols < min)] = min
            levels = np.linspace(min,max,ncont+1)
            if npoints:
                lvls = np.logspace(0,3,10)
                if not image: pl.contourf(xgri, ygri, num,norm=LogNorm(),levels=lvls)
                if image: pl.imshow(num, extent=(xgri[0,0], xgri[-1,0], ygri[0,0], ygri[0,-1]),interpolation='nearest')#, cmap=cm.gist_rainbow)
            else:
                print(ext)
                #pl.scatter(xgri[0,0],ygri[0,0],s=10)
                if image: pl.imshow(cols.T, origin='lower', extent=ext,interpolation='nearest')#, cmap=cm.gist_rainbow)
                if not image: pl.contourf(xgri, ygri, cols,levels=levels)
        #cs = pl.contour(xgri, ygri, cols,ncont)
        #pl.clabel(cs, inline=1,fontsize=10)#,manual=manual_locations)

    if form:
        pl.colorbar(format='%.1e')
    else:
        pl.colorbar()

    if addpoints:
        if(condpo is None):
            if(min == max):
                print(np.min(cols[~np.isnan(cols)]),np.max(cols[~np.isnan(cols)]))
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=np.min(cols[~np.isnan(cols)]),vmax=np.max(cols[~np.isnan(cols)]))
            else:
                sca = pl.scatter(x2,y2,c = col2,s=1,edgecolors='none',vmin=min,vmax=max)
        else:
            print('cisiamo')
            print(len(x[condpo]))
            print(x)
            print(condpo)
            print(len(condpo))
            sca = pl.scatter(x[condpo],y[condpo],color='black',marker='o',s=1)

    vip4x, vip4y = map(360-vip4_lon,vip4_lat)

    vip4x = np.append(vip4x,vip4x[0])
    vip4y = np.append(vip4y,vip4y[0])
    stat_x = np.append(stat_x,stat_x[0])
    stat_y = np.append(stat_y,stat_y[0])
    if aur_model == 'VIP4':
        pl.plot(vip4x,vip4y,color='white',linewidth=2.0)
        pl.plot(vip4x,vip4y,color='black',linewidth=2.0,linestyle='--')
    elif aur_model == 'stat':
        pl.plot(stat_x,stat_y,color='white',linewidth=2.0)
        pl.plot(stat_x,stat_y,color='black',linewidth=2.0,linestyle='--')
    else:
        pl.plot(vip4x,vip4y,color='white',linewidth=2.0)
        pl.plot(vip4x,vip4y,color='black',linewidth=2.0,linestyle=':')
        pl.plot(stat_x,stat_y,color='white',linewidth=2.0)
        pl.plot(stat_x,stat_y,color='black',linewidth=2.0,linestyle='--')

    if live: show=True
    if(show): pl.show()
    if not live:
        if nomefi is not None: fig.savefig(nomefi, format='eps', dpi=150)
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

    map.drawmeridians(np.arange(0,360,30),labels=[1,1,0,1],fontsize=10)

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
            stereoplot(lon,lat,col,nomefi=nomefi,polo=polo,min=vmin,max=vmax,condpo=cond,live=live,pdf=pdf)
        else:
            stereoplot(lon,lat,col,polo=polo,min=vmin,max=vmax,condpo=cond,live=live,pdf=pdf)

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
        filename = '/home/fede/Scrivania/Jiram/DATA/Model_VIP4/southR30table.txt'
    if polo == 'N':
        filename = '/home/fede/Scrivania/Jiram/DATA/Model_VIP4/northR30table.txt'

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
        filename = '/home/fede/Scrivania/Jiram/DATA/Model_VIP4/statistico/south_aurora.txt'
    if polo == 'N':
        filename = '/home/fede/Scrivania/Jiram/DATA/Model_VIP4/statistico/north_aurora.txt'

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


def leggi_console(polo,cartres=None):
    if polo == 'S':
        cart = '/home/fede/Scrivania/Jiram/DATA/JM0003_all/aur_S_nadir_70_CH4/'
        if cartres is None: cartres = cart
        lch4 = True
        l500 = True  # così fa tutto con le geometrie a 500 km
        lN80 = False #True # così aggiungo le nuove misure fino a 80 gradi di emission angle
    else:
        cart = '/home/fede/Scrivania/Jiram/DATA/JM0003_all/aur_N_nadir_70_CH4/'
        if cartres is None: cartres = cart
        lch4 = True
        l500 = True  # così fa tutto con le geometrie a 500 km
        lN80 = True #True # così aggiungo le nuove misure fino a 80 gradi di emission angle
        emissMAX = 80.0
    cart80 = '/home/fede/Scrivania/Jiram/DATA/JM0003_all/aur_N_nadir_80_CH4/'

    ######################################################################################à

    pixs = pickle.load(open(cart+'all_pixs.pic','r'))

    if l500:
        # si va a prendere le geometrie per 500 km
        p500 = pickle.load(open(cart+'all_pixs_500.pic','r'))
        pixs500 = p500.view(np.recarray)
        print(len(pixs),len(pixs500))
        print(pixs[1139])
        print(pixs500[1139])

    #sys.exit()
    print(type(pixs.cubo))
    cubs = np.unique(pixs.cubo)
    print(cubs)

    ii,col1, err_col = read_res_jir_3(cartres+'CD-H3p.dat')
    iit,temp1, err_temp = read_res_jir_3(cartres+'VT-H3p.dat')
    iic,chi1 = read_res_jir_4(cartres+'chisq.dat')
    iio,off1 = read_res_jir_4(cartres+'offset_ok.dat')
    iis,shi1 = read_res_jir_4(cartres+'shift_ok.dat')
    if lch4: ii4,ch41, err_ch4 = read_res_jir_3(cartres+'CD-CH4.dat')

    col = np.zeros(len(pixs))
    err_c = np.zeros(len(pixs))
    temp = np.zeros(len(pixs))
    err_t = np.zeros(len(pixs))
    chi = np.zeros(len(pixs))
    off = np.zeros(len(pixs))
    shi = np.zeros(len(pixs))
    if lch4: ch4 = np.zeros(len(pixs))

    col[ii] = col1
    err_c[ii] = err_col
    temp[iit] = temp1
    err_t[iit] = err_temp
    chi[iic] = chi1
    chi[chi == 0.0] = np.nan
    off[iio] = off1
    shi[iis] = shi1
    if lch4: ch4[ii4] = ch41

    if lN80:
        p80 = pickle.load(open(cart80+'all_pixs.pic','r'))
        ii,col1, err_col = read_res_jir_3(cart80+'CD-H3p.dat')
        iit,temp1, err_temp = read_res_jir_3(cart80+'VT-H3p.dat')
        iic,chi1 = read_res_jir_4(cart80+'chisq.dat')
        iio,off1 = read_res_jir_4(cart80+'offset_ok.dat')
        iis,shi1 = read_res_jir_4(cart80+'shift_ok.dat')
        if lch4: ii4,ch41, err_ch4 = read_res_jir_3(cart80+'CD-CH4.dat')

        col80 = np.zeros(len(p80))
        err_c80 = np.zeros(len(p80))
        temp80 = np.zeros(len(p80))
        err_t80 = np.zeros(len(p80))
        chi80 = np.zeros(len(p80))
        off80 = np.zeros(len(p80))
        shi80 = np.zeros(len(p80))
        if lch4: ch480 = np.zeros(len(p80))

        col80[ii] = col1
        err_c80[ii] = err_col
        temp80[iit] = temp1
        err_t80[iit] = err_temp
        chi80[iic] = chi1
        chi80[chi80 == 0.0] = np.nan
        off80[iio] = off1
        shi80[iis] = shi1
        if lch4: ch480[ii4] = ch41

        print(type(pixs),len(pixs))
        p80 = p80.view(np.recarray)
        co80 = (p80.emiss_angle < emissMAX)
        p80 = p80[co80]
        pixs = np.append(pixs,p80)
        pixs500 = np.append(pixs500,p80)
        print(type(pixs),len(pixs))
        pixs = pixs.view(np.recarray)
        pixs500 = pixs500.view(np.recarray)
        print(type(pixs),len(pixs))
        col = np.append(col,col80[co80])
        err_c = np.append(err_c,err_c80[co80])
        temp = np.append(temp,temp80[co80])
        err_t = np.append(err_t,err_t80[co80])
        chi = np.append(chi,chi80[co80])
        off = np.append(off,off80[co80])
        shi = np.append(shi,shi80[co80])
        if lch4: ch4 = np.append(ch4,ch480[co80])

    col_c = col*np.cos(np.pi*pixs.emiss_angle/180.0)
    err_cc = err_c*np.cos(np.pi*pixs.emiss_angle/180.0)

    cond2 = ((chi > 20) | (col <= 0.0) | (off < 0.0))

    col[cond2] = float('NaN')
    print(len(col[cond2]))
    temp[cond2] = float('NaN')
    chi[cond2] = float('NaN')
    off[cond2] = float('NaN')
    err_t[cond2] = float('NaN')
    err_c[cond2] = float('NaN')

    if lch4: ch4[cond2] = float('NaN')
    if lch4: ch4[(ch4<0.0)] = 0.0

    col_c = col*np.cos(np.pi*pixs.emiss_angle/180.0)
    err_cc = err_c*np.cos(np.pi*pixs.emiss_angle/180.0)
    if l500:
        col_c = col*np.cos(np.pi*pixs500.emiss_angle/180.0)
        err_cc = err_c*np.cos(np.pi*pixs500.emiss_angle/180.0)
    print(pixs.start_time[0])

    lon = pixs.pc_lon
    lat = pixs.pc_lat
    lab =''
    if l500:
        lon = pixs500.pc_lon
        lat = pixs500.pc_lat
        lab = '_500'

    if lN80:
        lab = '_lN80'

    if polo == 'N':
        mcol = 3e12
        npix = 40
    else:
        mcol = 4e12
        npix = 50

    return pixs,pixs500,col,col_c,err_c,temp,err_t,chi,off,shi,ch4


def scatt_color(x,y,file=None,c=None,xlabel='',ylabel='',title=''):
    """
    Makes a scatter plot with color scale.
    :return:
    """
    #
    return