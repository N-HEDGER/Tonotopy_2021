from utils import *
from base import *


class Plotter():
    
    """ Plotter
       Performs plotting based on subject output.

    """
    
    def __init__(self,subj,yaml_file):
        

        self.subj=subj
        
        self.yaml=yaml_file
        
        self._internalize_config_yaml()
        self.frame=pd.read_csv(self.subj.avcsv)
        
        self.flat_dir=self.subj.out_flat
        self.webGL_dir=self.subj.out_webGL
        self.sub=self.subj.subject
        
                
    def _internalize_config_yaml(self):
        
        """internalize_config_yaml
        Loads in the params for plotting the flatmaps.

        """
        
        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

        self.vis_dict = self.y['img_visualization']

        for key in self.vis_dict.keys():
            setattr(self, key, self.vis_dict[key])
            
            
    def saveout(self):
        
        """ saveout
        Saves out all the plots
        
        """
        
        for i in range(len(self.vars2plot)):
            if np.bool(self.zoom[i])==True:
                self.make_zoomplot(i)
            
            else:
    
                self.make_plot(i)        
    
    def make_plot(self,plotnum,alpha=False,save=True):
        
        """ make_plot
        Make a plot.
        
        """
        uber_plot(self.frame[self.vars2plot[plotnum]],subject=self.pycortex_subject,vmin=self.ranges[plotnum][0],vmax=self.ranges[plotnum][1],cmap=self.cmaps[plotnum],dat2=None,zoom=False,zoomrect=[-210.33542, -130.50809, -110.665405, 2],x=self.x,y=self.y,varname=self.plot_wildcard.format(subject=self.sub,param=self.vars2plot[plotnum]),save=True,path=self.flat_dir,with_rois=self.with_rois,with_colorbar=self.with_colorbar,with_curvature=self.with_curvature)
        
        
    def make_zoomplot(self,plotnum,alpha=False,save=True):
        
        """ make_plot
        Make a plot zoomed on an ROI
        
        """
        uber_plot(self.frame[self.vars2plot[plotnum]],subject=self.pycortex_subject,vmin=self.ranges[plotnum][0],vmax=self.ranges[plotnum][1],cmap=self.cmaps[plotnum],dat2=None,zoom=True,zoomrect=self.zoom_rectL,x=self.x,y=self.y,varname=self.zoomplot_wildcard.format(subject=self.sub,hem='L',param=self.vars2plot[plotnum]),save=True,path=self.flat_dir,with_rois=self.with_rois,with_colorbar=self.with_colorbar,with_curvature=self.with_curvature)
        uber_plot(self.frame[self.vars2plot[plotnum]],subject=self.pycortex_subject,vmin=self.ranges[plotnum][0],vmax=self.ranges[plotnum][1],cmap=self.cmaps[plotnum],dat2=None,zoom=True,zoomrect=self.zoom_rectR,x=self.x,y=self.y,varname=self.zoomplot_wildcard.format(subject=self.sub,hem='R',param=self.vars2plot[plotnum]),save=True,path=self.flat_dir,with_rois=self.with_rois,with_colorbar=self.with_colorbar,with_curvature=self.with_curvature)

        
class AggPlotter(Plotter):
    
    """ AggPlotter
       Performs plotting based on a csv file.
       Used for when this csv file is not associated with a subject class (i.e for data averaged across subjects)

    """
    
    def __init__(self,csv,yaml_file,out_dir='DEFAULT',sub='DEFAULT',vars2plot=['mu','sigma']):
        
        self.subj=os.path.split(csv)[1][:-4]
        
        if sub=='DEFAULT':
            self.sub=os.path.split(csv)[1][:-4]
        else: 
            self.sub=sub
        
        self.yaml=yaml_file
        
        self._internalize_config_yaml()
        self.frame=pd.read_csv(csv)
        
   
        if out_dir=='DEFAULT':
            self.flat_dir=os.path.join(os.path.split(os.path.split(csv)[0])[0],'flatmaps')
        else:
            self.flat_dir=out_dir
            
        self.vars2plot=self.vars2plotagg
        self.ranges=self.rangesagg
        self.cmaps=self.cmapsagg
        
        
        
    def make_zoomplot(self,plotnum,alpha=False,save=True):
        
        """ make_zoomplot
        Make a plot zoomed on an ROI
        
        """
        uber_plot(self.frame[self.vars2plot[plotnum]],subject=self.pycortex_subject,vmin=self.ranges[plotnum][0],vmax=self.ranges[plotnum][1],cmap=self.cmaps[plotnum],dat2=self.frame[self.alphavar],zoom=True,zoomrect=self.zoom_rectL,x=self.x,y=self.y,varname=self.zoomplot_wildcard.format(subject=self.sub,hem='L',param=self.vars2plot[plotnum]),save=True,path=self.flat_dir,vmax2=self.vmax2[plotnum],with_rois=self.with_rois,with_colorbar=self.with_colorbar,with_curvature=self.with_curvature)
        
        uber_plot(self.frame[self.vars2plot[plotnum]],subject=self.pycortex_subject,vmin=self.ranges[plotnum][0],vmax=self.ranges[plotnum][1],cmap=self.cmaps[plotnum],dat2=self.frame[self.alphavar],zoom=True,zoomrect=self.zoom_rectR,x=self.x,y=self.y,varname=self.zoomplot_wildcard.format(subject=self.sub,hem='R',param=self.vars2plot[plotnum]),save=True,path=self.flat_dir,vmax2=self.vmax2[plotnum],with_rois=self.with_rois,with_colorbar=self.with_colorbar,with_curvature=self.with_curvature)




def basic_plot(dat,vmax,subject='hcp_999999',vmin=0,rois=False,colorbar=False,cmap='plasma',ax=None,labels=True):
    
    """ basic_plot
    Plots 1D data using pycortex
        
    """
    
    dat=np.array(dat)
    
    light=cortex.Vertex(dat,subject=subject, vmin=vmin, vmax=vmax,cmap=cmap)
    mfig=cortex.quickshow(light,with_curvature=True,with_rois=rois,with_colorbar=colorbar,with_labels=labels,fig=ax)
    return mfig

def alpha_plot(dat,dat2,vmin,vmax,vmin2,vmax2,subject='hcp_999999',rois=False,labels=False,colorbar=False,cmap='nipy_spectral_alpha',ax=None):
    """ alpha_plot
    Plots 2D data using pycortex
        
    """
    
    light=cortex.Vertex2D(dat,dat2,subject=subject, vmin=vmin, vmax=vmax,vmin2=vmin2,vmax2=vmax2,cmap=cmap)
    mfig=cortex.quickshow(light,with_curvature=True,with_rois=rois,with_colorbar=colorbar,fig=ax,with_labels=labels)
    return mfig
    
    
def uber_plot(dat,subject,vmin,vmax,cmap,dat2=None,zoom=False,zoomrect=[-210.33542, -130.50809, -110.665405, 2],x=6,y=3,varname='variable',save=False,path='',vmin2=0,vmax2=1,**kwargs):
    
    
    """ uber_plot
    Flexible plotter of data with pycortex.
        
    """
    
    figure= plt.figure(figsize=(x,y))
    dat=np.array(dat)
    if dat2 is None:
        alpha=False
        dat2=np.ones_like(dat)
        vx=cortex.Vertex2D(dat,dat2,subject=subject, vmin=vmin, vmax=vmax,vmin2=0,vmax2=1,cmap=cmap)
    else:
        alpha=True
        cmap=cmap+'_alpha'
        vx=cortex.Vertex2D(dat,dat2,subject=subject, vmin=vmin, vmax=vmax,vmin2=vmin2,vmax2=vmax2,cmap=cmap)
        
    mfig=cortex.quickshow(vx,fig=figure,**kwargs)
    
    if zoom==True:
        plt.axis(zoomrect)
        
    fstring=os.path.join(path,''.join([varname+'_'+['','alpha'][int(alpha)],['full','zoom'][int(zoom)],'.png']))
    
    if save==True:
        mfig.savefig(fstring)
        
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()
            
    print(fstring)
    return mfig
    
    
def zoom_to_roi(subject, roi, hem, margin=15.0):
    
    """ zoom_to_roi
    zooms an roi specified in svg file for the subject.
        
    """
    
    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin
    
    
    plt.axis([xmin, xmax, ymin, ymax])
    print([xmin, xmax, ymin, ymax])
    return


def zoom_to_rect(myrect):
    
    """ zoom_to_rect
    zooms an arbitrary rectangle defined in the flatmap space.
        
    """
    
    plt.axis(myrect)




