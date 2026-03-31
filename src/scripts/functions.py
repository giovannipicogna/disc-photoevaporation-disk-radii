import os
import h5py as h
import numpy as np
from astropy import units as u
from astropy import constants as const
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from numpy import ndarray, sqrt


#############################################
# Functions to read PLUTO outputs
#############################################

def save_dict(di_, filename_):
    from six.moves import cPickle as pickle #for performance
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def getFilenames():
    filenames = [x for x in os.listdir("./") if ".h5" in x]
    filenames.sort()
    return filenames


def getVar(filename, variable):
    h5 = h.File(filename, "r")
    keys = [key for key in h5.keys()]
    step = int(keys[0][9:])
    returnData = h5["Timestep_" + str(step) + "/vars"][variable][:]
    h5.close()
    return returnData


def getGridCell(filename=None, all=1):
    if not (filename):
        filename = getFilenames()[0]
    h5 = h.File(filename, "r")
    if all:
        x = h5["cell_coords"]["X"][:]
        y = h5["cell_coords"]["Y"][:]
        z = h5["cell_coords"]["Z"][:]
    else:
        x = h5["cell_coords"]["X"]
        y = h5["cell_coords"]["Y"]
        z = h5["cell_coords"]["Z"]
    x = x.astype("float64")
    y = y.astype("float64")
    z = z.astype("float64")
    return x, y, z


def read_grid(output, rscale=10.*u.AU):

    grid = dict()

    xcell, ycell, zcell = (((getGridCell(filename=output)) * rscale).to(u.cm)).value

    grid['r'] = np.sqrt(xcell**2+ycell**2.+zcell**2)
    grid['th'] = np.arccos(ycell/grid['r'])

    grid['X'] = xcell
    grid['Z'] = ycell

    return grid


def read_data(output, grid, rscale=10., mscale=1., mu=1.37125):

    data = dict()

    rscale *= u.AU
    mscale *= u.solMass
    vscale = np.sqrt(const.G*mscale/rscale)/2.0/np.pi
    rhoscale = mscale/rscale**3
    pscale = vscale**2*rhoscale

    data['density'] = (getVar(output, 'rho') * rhoscale).to(u.g/u.cm**3)
    data['v_r'] = ((getVar(output, 'vx1') * vscale).to(u.cm/u.s)).value
    data['v_th'] = ((getVar(output, 'vx2') * vscale).to(u.cm/u.s)).value
    data['v_ph'] = ((getVar(output, 'vx3') * vscale).to(u.cm/u.s)).value
    data['pressure'] = (getVar(output, 'prs') * pscale).to(u.barye)
    data['column_density'] = getVar(output, 'cd')
    data['temperature_dust'] = getVar(output, 'Tdust')
    data['temperature_gas'] = ((data['pressure']*mu*const.m_p)/(const.k_B*data['density'])).to(u.K).value

    data['density'] = data['density'].value
    data['pressure'] = data['pressure'].value

    data['v_x'] = data['v_r']*np.sin(grid['th']) + data['v_th']*np.cos(grid['th'])
    data['v_z'] = data['v_r']*np.cos(grid['th']) - data['v_th']*np.sin(grid['th'])

    return data


def remove_disc(data, grid, Mstar, fin):
    """Remove the bound disk from the wind region.

    Parameters
    ----------
    data : dictionary
        A dictionary containing all the physical quantities in the grid
    grid : dictionary
        A dictionary containing the grid

    Returns
    -------
    data_cut: dictionary
        A dictionary containing all the physical quantities in the wind region
    th_cut: array
        Array with the theta index at each radius where the wind region starts
    r_hole: double
        Value of the cavity radius (in AU)
    """
    import copy

    DIMT = grid['X'].shape[0]

    v_gas = sqrt(data['v_r']**2 + data['v_th']**2 + data['v_ph']**2)
    rho = data['density']
    p = data['pressure']
    gamma = 1.4
    G = (const.G.cgs).value
    M = (Mstar*const.M_sun.cgs).value
    r = grid['r']
    bernoulli_constant = 0.5*v_gas*v_gas + gamma/(gamma-1)*p/rho - G*M/r

    data_cut = copy.deepcopy(data)

    # Carving out the disc from the wind
    for i in range(fin):
        for key in data.keys():
            data_cut[key][bernoulli_constant<=0] = -1

    th_cut = np.zeros(fin, dtype=int)

    # Select the theta index at each radius where the logarithm of
    # the chosen variable has a minimum
    for i in range(fin):
        th_cut[i] = np.argmax(bernoulli_constant[:,i]<=0)  # index of the first cell in the disc

    # dx_r_gap = np.max(np.where(th_cut==0))  # Find the location of the cavity
    # th_cut[:idx_r_gap] = DIMT-1    # Cut the region inside the cavity

    r_hole = grid["r"][0, 0]

    return data_cut, th_cut, r_hole


def remove_disc_var(var_cut, var, init):

    a = np.shape(var_cut)[1]
    lgvar = np.log10(var_cut)
    Th_cut = np.zeros(a, dtype=int)
    for i in range(init, a):
        dlgvar = np.diff(lgvar[:, i])
        index_cut = dlgvar.argmin()  # index of the first cell in the disc
        var[index_cut:, i] = -1
        Th_cut[i] = index_cut

    return var, Th_cut


def regrid(data, grid, res=4000):
    """Regrid the computational region with the given resolution.

    Parameters
    ----------
    data : dictionary
        A dictionary containing all the physical quantities in the grid
    grid : dictionary
        A dictionary containing the grid
    res : int
        Resolution for each dimension of the new grid

    Returns
    -------
    Func_density: array
        Array
    Func_vr: array
        Array
    Func_vx: array
        Array
    Func_vz: array
        Array
    """
    from scipy.interpolate import griddata, RegularGridInterpolator

    DIMT = grid['X'].shape[0]
    DIMR = grid['X'].shape[1]
    xcart = np.logspace(np.log10(1e12), np.log10(grid['r'].max()*0.99999), res)
    zcart = np.copy(xcart)
    Xc, Zc = np.meshgrid(xcart, zcart)

    density = griddata((grid['X'].reshape(DIMR*DIMT),
                       grid['Z'].reshape(DIMR*DIMT)),
                       data["density"].reshape(DIMR*DIMT),
                       (Xc.reshape(res**2), Zc.reshape(res**2)),
                       method='nearest', fill_value=0.)
    v_r = griddata((grid['X'].reshape(DIMR*DIMT),
                   grid['Z'].reshape(DIMR*DIMT)),
                   data["v_r"].reshape(DIMR*DIMT),
                   (Xc.reshape(res**2), Zc.reshape(res**2)),
                   method='nearest', fill_value=0.)
    v_x = griddata((grid['X'].reshape(DIMR*DIMT),
                   grid['Z'].reshape(DIMR*DIMT)),
                   data["v_x"].reshape(DIMR*DIMT),
                   (Xc.reshape(res**2), Zc.reshape(res**2)),
                   method='nearest', fill_value=0.)
    v_z = griddata((grid['X'].reshape(DIMR*DIMT),
                   grid['Z'].reshape(DIMR*DIMT)),
                   data["v_z"].reshape(DIMR*DIMT),
                   (Xc.reshape(res**2), Zc.reshape(res**2)),
                   method='nearest', fill_value=0.)

    density = density.reshape(res, res)
    v_r = v_r.reshape(res, res)
    v_x = v_x.reshape(res, res)
    v_z = v_z.reshape(res, res)

    Func_density = RegularGridInterpolator((xcart, zcart), density.T,
                                           bounds_error=False, fill_value=None)
    Func_vr = RegularGridInterpolator((xcart, zcart), v_r.T,
                                      bounds_error=False, fill_value=None)
    Func_vx = RegularGridInterpolator((xcart, zcart), v_x.T,
                                      bounds_error=False, fill_value=None)
    Func_vz = RegularGridInterpolator((xcart, zcart), v_z.T,
                                      bounds_error=False, fill_value=None)

    return Func_density, Func_vr, Func_vx, Func_vz


def calculate_mdot_wind(Th_cut, fin, Func_density, Func_vr, Func_vx, Func_vz,
                        grid):

    from scipy.integrate import cumulative_trapezoid as cumtrapz

    DIMT: int = grid['th'].shape[0]
    r_surf: np.float64 = grid['r'][DIMT-1, fin]
    rmax = grid['r'][DIMT-1]

    th_final: np.float64 = grid['th'][Th_cut[fin-1], fin]

    th_surf: np.ndarray[np.float64] = np.linspace(grid['th'][0, fin]*1.01, 0.99*th_final, DIMT-1)
    x_surf: np.ndarray[np.float64] = r_surf * np.sin(th_surf)
    z_surf: np.ndarray[np.float64] = r_surf * np.cos(th_surf)

    Rstart = []
    xsu: np.ndarray[np.float64] = np.zeros(len(th_surf))
    zsu: np.ndarray[np.float64] = np.zeros(len(th_surf))
    thsu: np.ndarray[np.float64] = np.zeros(len(th_surf))
    rsu: np.ndarray[np.float64] = np.zeros(len(th_surf))

    idx = 0
    for i in range(len(th_surf)):
        xs, zs = calc_streamline(rmax, x_surf[i], z_surf[i],
                                                       Func_vx, Func_vz)
        if (not (np.isnan(xs[-1])) and not (np.isnan(zs[-1]))):
            xsu[idx] = x_surf[i]
            zsu[idx] = z_surf[i]
            thsu[idx] = th_surf[i]
            rsu[idx] = r_surf
            Rstart.append(xs[-1])
            idx += 1

    sigma_dot_theta = Func_density((xsu, zsu)) * Func_vr((xsu, zsu))

    mask = sigma_dot_theta > 0.
    sigma_dot_theta = sigma_dot_theta[mask]
    thsu = thsu[mask]
    rsu = rsu[mask]

    integrand = 4.*np.pi*np.sin(thsu)*rsu**2.*sigma_dot_theta

    Mdot: ndarray[np.float64] = cumtrapz(integrand, thsu, initial=0.)

    idx: int = len(Rstart)

    return Rstart, Mdot, idx


def fit_mdot(Rstart, Mdot, j, hole_radius, plot=False,
             save=False, filename='data0110'):

    radii = []
    acc = []

    t = 0
    for i in range(j):
        spa = Rstart[i]*u.cm.to(u.AU)
        spb = Mdot[i]
        if (spa < hole_radius):
            continue
        else:
            radii.append(spa - hole_radius)
            acc.append(spb)
            t = t+1

    acc = acc/acc[-1]

    Mdarray = np.column_stack((radii, acc))
    bfg = Mdarray[Mdarray[:, 0].argsort()]

    bfg = bfg[~((bfg[:, 0] < 1e-2))]

    if save:
        np.save(filename, bfg)

    z = np.polyfit(np.log10(bfg[:, 0]), bfg[:, 1], 10)
    poly_mdot = np.poly1d(z)

    Mdot_val = (Mdot.max()*u.g/u.s).to(u.M_sun/u.yr)
    Sigma = np.gradient(np.log10(bfg[:, 1]), np.log10(bfg[:, 0]))
    zd = np.polyfit(bfg[:, 0], Sigma, 4)
    poly_sigma = np.poly1d(zd)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(bfg[:, 0], np.gradient(bfg[:, 1], np.log10(bfg[:, 0])) * Mdot_val, '.', label='sigma')
        plt.figure()
        plt.semilogx(bfg[:, 0], bfg[:, 1], '.', label='mdot')

    if save:
        np.savez(filename, x=bfg[:, 0], y=Sigma)

    return poly_mdot, poly_sigma


def calc_streamline(rmax, start_x, start_z, fvx, fvz):
    # fvx and fvz are functions that evaluate vx and vz at a given point.
    def diff_eqs(x, z):
        vz = fvz((x, z))
        vx = fvx((x, z))

        indomain = True

        if (x < 0 or z < 0):
            indomain = False
        if (abs(vz) < 1. and abs(vx) < 1.):
            #print(abs(vz), ((x*u.cm).to(u.au)).value, ((z*u.cm).to(u.au)).value)
            indomain = False
        if (x < 20 and z < 20):
            indomain = False

        return vz/vx, indomain

    xstream = []
    zstream = []
    xstream.append(start_x)
    zstream.append(start_z)

    in_domain = True
    counter = 0
    while in_domain:
        # calculate step-size based on local resolution
        radius = np.sqrt(xstream[-1]**2.+zstream[-1]**2.)
        i_r = (abs(rmax-radius)).argmin()
        # now calcuate dx, with correct sign
        dx = -np.sign(fvx((xstream[-1], zstream[-1])))*(rmax[i_r]-rmax[i_r-1])/10.
        # now calculate RK co-effients
        k1, in_domain = diff_eqs(xstream[-1], zstream[-1])
        k2, in_domain = diff_eqs(xstream[-1]+dx/2., zstream[-1]+dx/2.*k1)
        k3, in_domain = diff_eqs(xstream[-1]+dx/2., zstream[-1]+dx/2.*k2)
        k4, in_domain = diff_eqs(xstream[-1]+dx, zstream[-1]+dx*k3)

        if in_domain:
            zstream.append(zstream[-1]+dx/6.*(k1+2.*k2+2.*k3+k4))
            xstream.append(xstream[-1]+dx)
        #else:
        #    print(((zstream[-1]+dx/6.*(k1+2.*k2+2.*k3+k4))*u.cm).to(u.AU).value, ((xstream[-1]+dx)*u.cm).to(u.AU).value)


        counter += 1
        if (counter > 5000):
            break

    return xstream, zstream


def boxplot_2d(x, y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    # the box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec='k',
        zorder=0
    )
    ax.add_patch(box)

    # the x median
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    # the y median
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    # the central point
    ax.plot([xlimits[1]], [ylimits[1]], color='k', marker='o')

    # the x-whisker
    # defined as in matplotlib boxplot:
    # As a float, determines the reach of the whiskers to the beyond the
    # first and third quartiles. In other words, where IQR is the
    # interquartile range (Q3-Q1), the upper whisker will extend to
    # last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    # ##will extend to the first datum greater than Q1 - whis*IQR. Beyond
    # the whiskers, data are considered outliers and are plotted as
    # individual points. Set this to an unreasonably high value to force
    # the whiskers to show the min and max values. Alternatively, set this
    # to an ascending sequence of percentile (e.g., [5, 95]) to set the
    # whiskers at specific percentiles of the data. Finally, whis can
    # be the string 'range' to force the whiskers to the min and max of
    # the data.
    iqr = xlimits[2]-xlimits[0]

    # left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # the y-whisker
    iqr = ylimits[2]-ylimits[0]

    # bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [bottom, ylimits[0]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [bottom, bottom],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [top, ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [top, top],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    # outliers
    mask = (x < left) | (x > right) | (y < bottom) | (y > top)
    ax.scatter(
        x[mask], y[mask],
        facecolors='none', edgecolors='k'
    )
