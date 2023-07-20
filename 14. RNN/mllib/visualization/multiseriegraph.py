# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        PLOTS, GRAPHS AND VISUALIZATION HELPERS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D #, axes3d



# ----------------------------------------------------------------------
#      
# ----------------------------------------------------------------------
class GraphConsts():
    DEFAULT_WIDTH=528
    DEFAULT_HEIGHT=297

    DEFAULT_SCREEN_DPI=70
    DEFAULT_IMAGE_DPI=DEFAULT_SCREEN_DPI * 3
    


#------------------------------------------------------------------------------------
def MinMaxNormalize(p_oImg):
    #tMin = np.zeros(p_oImg.shape, p_oImg.dtype)
    #tMax = np.zeros(p_oImg.shape, p_oImg.dtype)                   
    
    tMin=np.min(p_oImg)
    tMax=np.max(p_oImg)
    tNewMax=np.maximum(tMax-tMin, 1e-4)
    
    oNormalizedImage=(p_oImg-tMin)/ tNewMax
    return oNormalizedImage
#------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------
def Standardize(p_oImg):
    #tMin = np.zeros(p_oImg.shape, p_oImg.dtype)
    #tMax = np.zeros(p_oImg.shape, p_oImg.dtype)                   
    
    tMean=np.mean(p_oImg)
    tStd=np.std(p_oImg)
    
    oNormalizedImage=(p_oImg-tMean)/ tStd
    return oNormalizedImage
#------------------------------------------------------------------------------------







#==================================================================================================
class GraphSetup(object):
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        self.Title = None
        self.CaptionX = "X"
        self.CaptionY =  "Y"
        self.IsShowingGrid = False
        self.PixelsX = 1024
        self.PixelsY = 768
        self.DPI = GraphConsts.DEFAULT_IMAGE_DPI
        self.LegendFontSize = 11
        self.CommonLineWidth=1
        self.CommonColor="#0000f0"
        self.DisplayFinalValue=False
        self.XLimit=None
        self.YLimit=None     
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================




        
#==================================================================================================
class CMultiSerieGraph(object):
    __verboseLevel = 0
    
    DEFAULT_SERIE_COLORS = [
         '#4dbeee' # light-blue
        ,'#77ac30' # green
        ,'#d95319' # orange            
        ,'#7e2f8e' # purple
        ,'#edb120' # yellow            
        ,'#0072bd' # blue            
        ,'#a2142f' # red              
        ];
            
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        self.Setup = GraphSetup()
        self.NumberOfPoints = 0
        self.XValues = None
        self.YValues = None
        self.Labels=None        
        self.Colors=None
        self.Widths=None
        self.Styles=None
        self.PointsOfInterest=None
        #................................................................................
    #------------------------------------------------------------------------------------
    def Initialize(self, p_nX, p_nY, p_oLabels=None, p_oColors=None, p_oWidths=None, p_oStyles=None, p_oPointsOfInterest=None):
        self.XValues = p_nX
        self.YValues = p_nY
        self.NumberOfPoints = len(self.YValues)
        self.PointsOfInterest = p_oPointsOfInterest

        if type(self).__verboseLevel >=1 :
            if p_nX.shape[0] != p_nY[0].shape:
                print("Different number of points between X and Y", p_nX.shape, p_nY[0].shape)
        
        #assert p_nX.shape[0] == p_nY.shape[0], "Different number of points between X and Y"
        
        if p_oLabels is not None:   
            self.Labels=p_oLabels 
                            
        self.Colors=[]
        self.Widths=[]
        self.Styles=[]

        for nIndex in range(0, self.NumberOfPoints):
            self.Colors.append(self.Setup.CommonColor)
            self.Widths.append(self.Setup.CommonLineWidth)
            self.Styles.append("-")
        


        
        if p_oColors is not None:
            self.Colors = p_oColors
        else:
            self.Colors = type(self).DEFAULT_SERIE_COLORS
        if p_oWidths is not None:
            self.Widths = p_oWidths
        if p_oStyles is not None:
            self.Styles = p_oStyles
    #------------------------------------------------------------------------------------
    def Plot(self, p_sFileName=None, p_nDPI=GraphConsts.DEFAULT_IMAGE_DPI):
        if p_sFileName is None:
            plt.show()
        else:  
            plt.savefig(p_sFileName, dpi=p_nDPI)
    #------------------------------------------------------------------------------------
    def Render(self, p_oPixelsX=None, p_oPixelsY=None, p_bIsScatter=False, p_nXTicks=None, p_nYTicks=None, p_sLegendLocation="best", p_bIsMinMaxNormalized=False, p_bIsStandardized=False):
        if p_oPixelsX is not None:
            self.Setup.PixelsX = p_oPixelsX
        if p_oPixelsY is not None:
            self.Setup.PixelsY = p_oPixelsY
        
        fig = plt.figure(dpi=self.Setup.DPI)
        #fig = plt.figure(figsize=((p_nPixelsX/p_nDpi)*1.4, p_nPixelsY/p_nDpi), dpi=p_nDpi)
        ax = fig.add_subplot(1,1,1)
        
        if self.Setup.Title is not None:
            #GLOBAL TITLE:
            #plt.suptitle(self.Setup.Title)
            ax.title.set_text(self.Setup.Title)
        
        # Automatically picks a color
        #plt.plot(p_nX, p_nY, p_nSeriesStyles, label=self.SeriesLabels)
            
        # Uses user-defined color
        
        #PANTELIs [2017-06-18] BF: Disabled support for non-list
#         if len(self.YValues) == 1:
#             if p_bIsScatter:
#                 plt.scatter(self.XValues, self.YValues, self.SeriesStyles, linewidth=self.SeriesWidths, label=self.SeriesLabels, color=self.SeriesColor)
#             else:
#                 plt.plot(self.XValues, self.YValues, self.SeriesStyles, linewidth=self.SeriesWidths, label=self.SeriesLabels, color=self.SeriesColor)
#         else:
        for nIndex, nSerieY in enumerate(self.YValues):
            if p_bIsStandardized:
              nSerieY = Standardize(nSerieY)
              
            if p_bIsMinMaxNormalized:
              nSerieY = MinMaxNormalize(nSerieY)
          
            
          
            #print("Serie #%i" % nIndex)
            #print(len(self.SeriesColors), len(self.SeriesWidths), len(self.SeriesLabels))
            #print(self.Styles)
            if self.Labels is None:
                sLabel=None
            else:
                sLabel=self.Labels[nIndex]
            if p_bIsScatter:
                plt.scatter(self.XValues, nSerieY, color=self.Colors[nIndex], linewidth=self.Widths[nIndex], label=sLabel, linestyle=self.Styles[nIndex])
            else:
                plt.plot(self.XValues, nSerieY, color=self.Colors[nIndex], linewidth=self.Widths[nIndex], label=sLabel, linestyle=self.Styles[nIndex])
            
            if self.PointsOfInterest is not None:
              plt.plot(self.PointsOfInterest, nSerieY[self.PointsOfInterest],'or')
                      
                
        plt.xlabel(self.Setup.CaptionX, fontsize=12)
        plt.ylabel(self.Setup.CaptionY, fontsize=12)
    
                
        #plt.xticks(nTicks, 100)
    
        if self.Labels is not None:
            plt.legend(loc=p_sLegendLocation,prop={'size':self.Setup.LegendFontSize})
                
        plt.grid(self.Setup.IsShowingGrid)
        if True: #TEMP: TODO SUPPORT
            # major ticks every 20, minor ticks every 5                                      
            
            
            if p_nXTicks is not None:
                major_ticks_x = np.arange(0, p_nXTicks[0], p_nXTicks[1])                                              
                minor_ticks_x = np.arange(0, p_nXTicks[0], p_nXTicks[2])
                #ax.set_aspect(1.0)
                ax.set_xticks(major_ticks_x)                                                       
                ax.set_xticks(minor_ticks_x, minor=True)                                           
            
            if p_nYTicks is not None:
                major_ticks_y = np.arange(0, p_nYTicks[0], p_nYTicks[1])                                              
                minor_ticks_y = np.arange(0, p_nYTicks[0], p_nYTicks[2])                                               
                ax.set_yticks(major_ticks_y)                                                       
                ax.set_yticks(minor_ticks_y, minor=True)                                           
                                                           
            #ax.set_aspect(1.0)
#             ax.set_xticks(major_ticks_x)                                                       
#             ax.set_xticks(minor_ticks_x, minor=True)                                           
#             ax.set_yticks(major_ticks_y)                                                       
#             ax.set_yticks(minor_ticks_y, minor=True)                                           
            # and a corresponding grid                                                       
            
            plt.grid(which='both')                                                            
            
            # or if you want differnet settings for the grids:                               
            plt.grid(which='minor', alpha=0.1)                                                
            plt.grid(which='major', alpha=0.4)
        else:
            plt.xticks(np.arange(min(self.XValues), max(self.XValues)+1, 5))
            
            
            
            
        if self.Setup.XLimit is not None:
            ax.set_xlim(self.Setup.XLimit)
        if self.Setup.YLimit is not None:
            ax.set_ylim(self.Setup.YLimit)
            
            
            
        oXScales=[0]*len(self.YValues)
        oYScales=[0]*len(self.YValues)
        oYValues=[0]*len(self.YValues)
        if self.Setup.DisplayFinalValue:
            for nSerieIndex, nSerieY in enumerate(self.YValues):
                yvalueprev=None

                bIsEarlyBreak=False
                for nIndex in range(0,nSerieY.shape[0]):
                    yvalue = nSerieY[nIndex]
                    if (yvalue is None) or math.isnan(yvalue):
                        y_rescale,nScaleCurrent = self.__rescale(yvalueprev, oYScales, oYValues)
                        x_rescale=nIndex-2
                        if x_rescale in oXScales:
                            x_rescale += 6
                        oXScales[0]=x_rescale
                        oYScales[nSerieIndex] = nScaleCurrent
                        oYValues[nSerieIndex] = yvalueprev
                        plt.text(x_rescale, y_rescale, "%.2f" % yvalueprev, fontdict=None, withdash=False, fontsize=8)
                        bIsEarlyBreak=True
                        break
                    yvalueprev = yvalue
                
                if not bIsEarlyBreak:
                    x_rescale= nSerieY.shape[0]-2
                    if x_rescale in oXScales:
                        x_rescale  += 6                    
                    y_rescale,_ = self.__rescale(yvalueprev, oYScales, oYValues)
                    #plt.text(x_rescale, y_rescale, "%.2f" % yvalueprev, fontdict=None, withdash=False, fontsize=8)
                    plt.text(x_rescale, y_rescale, "%.2f" % yvalueprev, fontdict=None, fontsize=8)
                        
        
        fig.tight_layout()
    #------------------------------------------------------------------------------------
    def __rescale(self, yvalueprev, oYScales, oYValues, a_rescaler=0.01, a_scale_inc = 1):
        nScaleCurrent = int(np.round(yvalueprev / a_rescaler, 0))
        nResult = nScaleCurrent * a_rescaler
        
        for nPos, nScale in enumerate(oYScales):
            nValue=oYValues[nPos]
            if nScaleCurrent == nScale:
                if yvalueprev > nValue:
                    nScaleCurrent += a_scale_inc
                elif yvalueprev < nValue:
                    nScaleCurrent -= a_scale_inc

                nResult = nScaleCurrent * a_rescaler                                        
        
        assert nResult is not None
        return nResult, nScaleCurrent
    #------------------------------------------------------------------------------------

            
#==================================================================================================    









#------------------------------------------------------------------------------------
def ShowImageRGB(p_oImg, p_bIsInterpolating=False, p_sWindowTitle=None):
    VERBOSE_THIS=True  
#     tMin = np.zeros(p_oImg.shape, p_oImg.dtype)
#     tMax = np.zeros(p_oImg.shape, p_oImg.dtype)                   
#     
#     tMin=np.min(p_oImg)
#     tMax=np.max(p_oImg)
#     tNewMax=np.maximum(tMax-tMin, 1e-4)
    
    #oNormalizedImage=(p_oImg-tMin)/ tNewMax
    
    if VERBOSE_THIS:
        print("IMAGE SHAPE", p_oImg.shape)
    
    oNormalizedImage=MinMaxNormalize(p_oImg)
    
    ShowImage(oNormalizedImage, p_sWindowTitle)
#------------------------------------------------------------------------------------    
def ShowImage(p_oImg, p_sWindowTitle=None, p_bIsInterpolating=False):
    fig = plt.figure() 
    if p_bIsInterpolating:
        plt.imshow(p_oImg, vmin=np.min(p_oImg[:,:]), vmax=np.max(p_oImg[:,:]))
    else:
        plt.imshow(p_oImg, interpolation='none',  vmin=np.min(p_oImg[:,:]), vmax=np.max(p_oImg[:,:]))
    if p_sWindowTitle is not None:
        fig.canvas.set_window_title(p_sWindowTitle)    
    plt.set_cmap("jet")
    plt.show()
    
#------------------------------------------------------------------------------------
def ShowImageListRGB(p_oImageList,  p_sWindowTitle=None, p_nColumnsPerRow=3, p_sVerticalText=None, p_sHorizontalText=None, p_bIsTightLayout=True, p_nTightPad=3.5, p_nFigWidth=14, p_nFigHeight=None, p_bIsNormalizing=True, p_sColorMap="jet", p_bIsUsingAxis=False, p_nSquareAspectPixels=None, p_sSaveFileNameFull=None, p_sFaceColor=None):
    VERBOSE_THIS=True    
    

    #plt.clf()
    nImageCount = len(p_oImageList)
    nRows = int(np.ceil (nImageCount / p_nColumnsPerRow ))
    # Ensure one row
    nRows = np.maximum(nRows,1)
    if VERBOSE_THIS:
        print("Ploting %i images with %i rows of %i columns each" % (nImageCount, nRows, p_nColumnsPerRow))
    
    if p_nFigHeight is None:
        nFigHeight = np.ceil(30/p_nFigWidth) * nRows 
    else:
        nFigHeight = p_nFigHeight
    
    if not p_bIsUsingAxis:
        #if p_sFaceColor is not None: 
        #    fig = plt.figure(figsize=(p_nFigWidth,nFigHeight),facecolor=p_sFaceColor)
        #else:
        fig = plt.figure(figsize=(p_nFigWidth,nFigHeight))
                
        gs = GridSpec(nRows, p_nColumnsPerRow)

        gs.update(wspace=0.005, hspace=0.05) # set the spacing between axes.        
    else:
        fig, gs = plt.subplots(nrows=nRows, ncols=p_nColumnsPerRow, sharex=True, sharey=True, figsize=(p_nFigWidth, nFigHeight))
    
    if p_sFaceColor is not None:   
        fig.set_facecolor(p_sFaceColor)
            

    
    # The map should be set after calling plt.subplots(), otherwise two plots will be generated
    if p_sColorMap is not None:
        plt.set_cmap(p_sColorMap)
    
    if p_sWindowTitle is not None: 
        plt.suptitle(p_sWindowTitle)
    
    # Turns axis off
    #if not p_bIsUsingAxis:    
    #    fig.patch.set_visible(False)
        
    #fig.text(0.5, 0.02 , "Frequency [Hz]", ha="center")
    if p_sVerticalText is not None:
        #fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
        fig.text(0.02, 0.5,  p_sVerticalText, va="center", rotation="vertical")

    if p_sHorizontalText is not None:
        #fig.text(0.5, 0.04, 'common X', ha='center')
        fig.text(0.35, 0.02,  p_sHorizontalText, va="center")


            
    assert len(p_oImageList) > 0, "image list is empty"
    nImageTuple = p_oImageList[0]
    bHasTitle = (len(nImageTuple) == 2)
    
    
    for nPlotColumnIndex, nImage in enumerate(p_oImageList[:]):
        if nRows==1:
            nRowIndex=0
            nColIndex=0
            axis=gs[nPlotColumnIndex]
#             subplot = plt.subplot(gs[nPlotColumnIndex])
#             if VERBOSE_THIS:
#                 print("Ploting column %i" % nPlotColumnIndex)
        else:
            nRowIndex = int(nPlotColumnIndex / p_nColumnsPerRow)
            nColIndex = (nPlotColumnIndex - nRowIndex * p_nColumnsPerRow)
            axis=gs[nRowIndex, nColIndex]
#               
#             subplot=plt.subplot(gs[nRowIndex, nColIndex])
#             if VERBOSE_THIS:
#                 print("Ploting column %i at (%i, %i)" % (nPlotColumnIndex, nRowIndex, nColIndex))
                
        if p_sFaceColor is not None:   
            axis.set_facecolor(p_sFaceColor)
                        
        subplot=plt.subplot(axis)
        if VERBOSE_THIS:
            print("Ploting column %i at (%i, %i)" % (nPlotColumnIndex, nRowIndex, nColIndex))

        
        if not p_bIsUsingAxis: 
            subplot.axis('off')
            subplot.set_xticklabels([])
            subplot.set_yticklabels([])
        subplot.set_aspect('equal')
        
        

        if nImage[0] is not None:
            subplot.set_xlim([0, nImage[0].shape[0]])
            subplot.set_ylim([0, nImage[0].shape[1]])
            
            if bHasTitle:
                if p_bIsNormalizing:
                    nImageNormalized=MinMaxNormalize(nImage[0])
                else:
                    nImageNormalized = nImage[0]
                sTitle = nImage[1]
            else:
                if p_bIsNormalizing:
                    nImageNormalized=MinMaxNormalize(nImage)
                else:
                    nImageNormalized = nImage
                sTitle = "image"

            #subplot.imshow(nImageNormalized, interpolation='none',  vmin=np.min(nImageNormalized[:,:]), vmax=np.max(nImageNormalized[:,:]))
            if p_nSquareAspectPixels is None:
                subplot.imshow(nImageNormalized, interpolation='none')
            else:
                subplot.imshow(nImageNormalized, interpolation='none' , extent=[0,p_nSquareAspectPixels,0,p_nSquareAspectPixels], aspect=1.0)
            
            
            subplot.margins(0)
        else:
            nImageNormalized = nImage[0]
            sTitle=""
                    
        #subplot.plot(p_nTicks, nFilteredSignal  , label=p_sFilterNames[nPlotColumnIndex], color=self.SignalColorDict[nRowIndex + 4])
        subplot.set_xlabel((sTitle))
        #subplot.grid(True)
        #subplot.legend(loc="lower left",prop={'size':8} )

    if p_bIsTightLayout and p_bIsUsingAxis:
        plt.tight_layout(pad=p_nTightPad, h_pad=0.0, w_pad=0.0)
        
    if p_sSaveFileNameFull is None:
        plt.show()
    else:
        plt.savefig(p_sSaveFileNameFull, dpi=GraphConsts.DEFAULT_SCREEN_DPI)
#------------------------------------------------------------------------------------
def SubplotImageAuto(p_oSubplot, p_sTitle, p_oImg, p_bIsGrays=False, p_bIsClearingTicks=False, p_sRowTitle=None):
    Range=( np.min(p_oImg[:,:]), np.max(p_oImg[:,:]) )
    if p_bIsClearingTicks:
        p_oSubplot.set_xticklabels([])
        p_oSubplot.set_yticklabels([])
        #p_oSubplot.set_aspect('equal')
                    
    SubplotImage(p_oSubplot, p_sTitle, p_oImg, Range, p_sRowTitle=p_sRowTitle)
#------------------------------------------------------------------------------------
def SubplotImage(p_oSubplot, p_sTitle, p_oImg, p_tRange = (0,1), p_nIsGrays=False, p_sRowTitle=None):
    #PANTELIS [2017-05-23]: p_oSubplot.set_title(p_sTitle)
    p_oSubplot.set_xlabel(p_sTitle)
    if (p_sRowTitle is not None):
        p_oSubplot.set_ylabel(p_sRowTitle)
    

    p_oSubplot.set_xticklabels([])
    p_oSubplot.set_yticklabels([])
    p_oSubplot.set_aspect('equal')


    if p_nIsGrays:
        imgplot=p_oSubplot.imshow(p_oImg, interpolation='none', cmap='Greys_r')
    else:
        imgplot=p_oSubplot.imshow(p_oImg, interpolation='none')
    
    imgplot.set_clim(p_tRange[0],p_tRange[1])    
#------------------------------------------------------------------------------------
def VisualizeReceptiveFieldWeights(data, p_bIs3D=False, p_nAzimuth=-45, p_nElevation=45, p_sFileNameToSave=None):
    DEFAULT_VISUALIZATION_FOLDER = "E:\\Thesis\\Papers\\=Images=\\ARNN\\"
        
    # Set up grid and test data
    #x=range(data.shape[0])
    #y=range(data.shape[1])
    #hf = plt.figure()
    #ha = hf.add_subplot(111, projection='3d')
    #X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    #ha.plot_surface(X, Y, data)
    #plt.show()
    if p_bIs3D:
        x = np.arange(0,data.shape[0],0.5)
        y = np.arange(0,data.shape[1],0.5)
        lx=len(x)
        ly=len(y)
        
        xs, ys = np.meshgrid(x, y)
        zs=np.zeros((ly,lx))
        for i in range(0,ly):
            for j in range(0,lx):
                zs[i,j]=data[int(np.floor(i/2)),int(np.floor(j/2))]
                
        
        #zs = xs**2 + ys**2
        
        fig = plt.figure()
        ax = Axes3D(fig, azim=p_nAzimuth, elev=p_nElevation)
        """
          *azim*             Azimuthal viewing angle (default -60)
          *elev*             Elevation viewing angle (default 30)
        """
        
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='jet')
    else:
        plt.imshow(data, interpolation='none')
    plt.set_cmap("jet")        
    if p_sFileNameToSave is not None:
        plt.savefig(DEFAULT_VISUALIZATION_FOLDER + p_sFileNameToSave)
    else:
        plt.show()
#------------------------------------------------------------------------------------  

    
    