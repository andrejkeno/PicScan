using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.Media.Capture;
using Windows.Media.MediaProperties;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.Graphics.Display;
using Windows.UI.Xaml.Media.Imaging;
using System.Threading.Tasks;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;


// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=391641

namespace PicScan
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        MediaCapture captureManager;        
        ImageEncodingProperties imageFormat;
        StorageFolder folder;
        WriteableBitmap bitmap;
        StorageFile file;
        ViewType value;
        bool isRectangle = false;

        Image<Bgr, Byte> imgNormal;
        Image<Bgr, Byte> finishedImage;
        Image<Bgr, Byte> imgRect;
        Image<Bgr, Byte> imgGaussian;
        UMat cannyEdges;
        UMat cannyEdgesDown;
        UMat grayUmat;
        UMat contoursUmat;
        UMat pyrDownUmat;
        UMat pyrUpUmat;      

        enum ViewType
        {
            ViewNormal,
            ViewGaussian,
            ViewGray,
            ViewPyrDown,
            ViewPyrUp,
            ViewCanny,
            ViewCannyDown,
            ViewContours,
            ViewRect,
            ViewFinished
        };
                
       
        public MainPage()
        {
            this.InitializeComponent();
            this.NavigationCacheMode = NavigationCacheMode.Required;            
        }
       
        protected override void OnNavigatedTo(NavigationEventArgs e)
        {
            StartCamera();                              
        }

        async private void StartCamera()
        {
            captureManager = new MediaCapture();
            await captureManager.InitializeAsync();
            capturePreview.Source = captureManager;
            await captureManager.StartPreviewAsync();
        }

        private async void StopCamera()
        {
            await captureManager.StopPreviewAsync();            
        }

        private async void captureButton_Click(object sender, RoutedEventArgs e)
        {

            file = await TakePicture();            

            StopCamera();

            ProcessImage(file);                  
                       

        }

        private void ProcessImage(StorageFile file)
        {
            value = ViewType.ViewFinished;
            imgNormal = new Image<Bgr, byte>(file.Path);
            finishedImage = new Image<Bgr, byte>(imgNormal.Width, imgNormal.Height);
            imgGaussian = imgNormal.Copy();               
           
            imgGaussian.Resize(400, 400, Inter.Linear, true);        
            imgRect = imgGaussian.Copy();
            imgGaussian.SmoothGaussian(3);                   

            grayUmat = new UMat();

            //Convert the image to grascale            
            CvInvoke.CvtColor(imgGaussian, grayUmat, ColorConversion.Bgr2Gray);            
            
            //Use image pyr to remove noise
            pyrDownUmat = new UMat();
            pyrUpUmat = new UMat();
            CvInvoke.PyrDown(grayUmat, pyrDownUmat);                   
            CvInvoke.PyrUp(pyrDownUmat, pyrUpUmat);     
                           
            //Canny and edge detection      
            cannyEdges = new UMat();
            cannyEdgesDown = new UMat();

            CvInvoke.Canny(pyrUpUmat, cannyEdges, 75, 200, 3, false);
            CvInvoke.Canny(pyrDownUmat, cannyEdgesDown, 75, 200, 3, false);
                        
            LineSegment2D[] line = CvInvoke.HoughLinesP(
                cannyEdgesDown,
                1, // Distance resolution in pixel related units
                Math.PI/45.0, // Angle resolution measured in radians
                20, // threshold
                30, // minimum line width
                10); // gap between lines                       

            List<RotatedRect> boxList = new List<RotatedRect>();

            contoursUmat = cannyEdgesDown.Clone();           
                       
            //Find countours
            using (  VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint() )
            {
                CvInvoke.FindContours(contoursUmat, contours, null, RetrType.List,
                    ChainApproxMethod.ChainApproxSimple);


                int count = contours.Size;

                for (int i = 0; i < count; i++)
                {
                    using (VectorOfPoint contour = contours[i])
                    using (VectorOfPoint approxContour = new VectorOfPoint())
                    {                        
                        CvInvoke.ApproxPolyDP(contour, approxContour,
                            CvInvoke.ArcLength(contour, true) * 0.05, true);
                        if (CvInvoke.ContourArea(approxContour, false) > 20000)
                        {
                            if (approxContour.Size == 4)
                            {                                
                                isRectangle = true;
                                System.Drawing.Point[] pts = approxContour.ToArray();
                                LineSegment2D[] edges = Emgu.CV.PointCollection.PolyLine(pts, true);

                                for (int j = 0; j < edges.Length; j++)
                                {
                                    double angle = Math.Abs(edges[(j + 1) % edges.Length]
                                        .GetExteriorAngleDegree(edges[j]));

                                    if (angle < 80 || angle > 100)
                                    {
                                        isRectangle = false;
                                        break;                                        
                                    }                                        
                                }

                                if (isRectangle)
                                {
                                    VectorOfPointF sortedContour = FindCenterAndCorners(approxContour);                                 
                                    
                                    boxList.Add(CvInvoke.MinAreaRect(approxContour));                               

                                    //Apply perspective transform    

                                    System.Drawing.PointF[] pointsrc = new System.Drawing.PointF[4];
                                    pointsrc[0] = new System.Drawing.PointF(sortedContour[0].X * 2, sortedContour[0].Y * 2);
                                    pointsrc[1] = new System.Drawing.PointF(sortedContour[1].X * 2, sortedContour[1].Y * 2);
                                    pointsrc[2] = new System.Drawing.PointF(sortedContour[2].X * 2, sortedContour[2].Y * 2);
                                    pointsrc[3] = new System.Drawing.PointF(sortedContour[3].X * 2, sortedContour[3].Y * 2);
                                   
                                    System.Drawing.PointF[] pointdst = new System.Drawing.PointF[4];
                                    pointdst[0] = new System.Drawing.PointF(0, 0);
                                    pointdst[1] = new System.Drawing.PointF(finishedImage.Width, 0);
                                    pointdst[2] = new System.Drawing.PointF(finishedImage.Width, finishedImage.Height);
                                    pointdst[3] = new System.Drawing.PointF(0, finishedImage.Height);
                                                                 
                                    Mat trans = CvInvoke.GetPerspectiveTransform(pointsrc, pointdst);
                                    CvInvoke.WarpPerspective(imgNormal, finishedImage, trans,
                                    finishedImage.Size, Inter.Linear, Warp.Default,
                                    BorderType.Constant, new MCvScalar(0));         
                                                                       
                                    break;
                                }                        
                            }
                            
                        }
                    }
                }
            }

            if (isRectangle)
            {
                //Draw rectangles
                foreach (RotatedRect box in boxList)
                {
                    imgRect.Draw(box, new Bgr(0, 255, 0), 3);
                }      
            }        

            switch (value)
            {
                case ViewType.ViewNormal:
                    ShowPicture(imgNormal);
                    break;

                case ViewType.ViewGaussian:
                    ShowPicture(imgGaussian);     
                    break;

                case ViewType.ViewGray:
                    ShowPicture(grayUmat);
                    break;

                case ViewType.ViewPyrDown:
                    ShowPicture(pyrDownUmat);
                    break;

                case ViewType.ViewPyrUp:
                    ShowPicture(pyrUpUmat);
                    break;

                case ViewType.ViewCanny:
                    ShowPicture(cannyEdges);                    
                    break;

                case ViewType.ViewCannyDown:
                    ShowPicture(cannyEdgesDown);
                    break;
                    
                case ViewType.ViewContours:
                    ShowPicture(contoursUmat);
                    break;

                case ViewType.ViewRect:
                    ShowPicture(imgRect);                    
                    break;

                case ViewType.ViewFinished:
                    if (isRectangle)
                        ShowPicture(finishedImage);
                    else
                        ShowPicture(imgNormal);
                    break;

                default:
                    if (isRectangle)
                        ShowPicture(finishedImage);
                    else
                        ShowPicture(imgNormal);      
                    break;
            }  
                      
                        
        }

        private VectorOfPointF FindCenterAndCorners(VectorOfPoint contour)
        {
            //Find center and corners of rectangle
            System.Drawing.PointF[] center = new System.Drawing.PointF[1];
            center[0] = new System.Drawing.PointF(0, 0);

            for (int c = 0; c < contour.Size; c++)
            {
                center[0].X += contour[c].X;
                center[0].Y += contour[c].Y;
            }
            center[0].X *= (float)(1.0 / contour.Size);
            center[0].Y *= (float)(1.0 / contour.Size);
            return sortCorners(contour, center);
        }

        private VectorOfPointF sortCorners(VectorOfPoint contour, System.Drawing.PointF[] center)
        {
            VectorOfPointF topCorners = new VectorOfPointF();
            VectorOfPointF bottomCorners = new VectorOfPointF();  
            System.Drawing.PointF[] corner = new System.Drawing.PointF[1];            

            for (int i = 0; i < contour.Size; i++)
            {
                corner[0] = new System.Drawing.Point((int) contour[i].X, contour[i].Y);

                if (contour[i].Y < center[0].Y)                
                    topCorners.Push(corner);                    
                
                else
                    bottomCorners.Push(corner);
            }            

            System.Drawing.PointF[] topLeft = new System.Drawing.PointF[1];
            topLeft[0] = topCorners[0].X > topCorners[1].X ? topCorners[1] : topCorners[0];

            System.Drawing.PointF[] topRight = new System.Drawing.PointF[1];
            topRight[0] = topCorners[0].X > topCorners[1].X ? topCorners[0] : topCorners[1];

            System.Drawing.PointF[] bottomLeft = new System.Drawing.PointF[1];
            bottomLeft[0] = bottomCorners[0].X > bottomCorners[1].X ? bottomCorners[1] : bottomCorners[0];

            System.Drawing.PointF[] bottomRight = new System.Drawing.PointF[1];
            bottomRight[0] = bottomCorners[0].X > bottomCorners[0].Y ? bottomCorners[0] : bottomCorners[1];
            VectorOfPointF sortedContour = new VectorOfPointF();

            if ( (topRight[0].X - topLeft[0].X) >= (bottomLeft[0].Y - topLeft[0].Y) )
            {
                sortedContour.Push(topLeft);           
                sortedContour.Push(topRight);
                sortedContour.Push(bottomRight);
                sortedContour.Push(bottomLeft);
            }
            else
            {
                sortedContour.Push(topRight);
                sortedContour.Push(bottomRight);
                sortedContour.Push(bottomLeft);
                sortedContour.Push(topLeft);
            }

            return sortedContour;
        }

        private void ShowPicture(Image<Bgr, Byte> image)
        {
            Image<Bgr, Byte> asd = image.Copy();
            asd.Save(file.Path);
            bitmap = asd.ToWriteableBitmap();
            bitmap.Invalidate();

            asd.Dispose();
            asd = null;         
           
            //Show photo
            imagePreview.Source = bitmap;
            bitmap = null;         


            ScanButton.Visibility = Visibility.Collapsed;
            Back.Visibility = Visibility.Visible;
            Save.Visibility = Visibility.Visible;
            Delete.Visibility = Visibility.Visible;
            toggleOptions.IsEnabled = true;  
        
            
            
        }

        

        private void ShowPicture(UMat umat)
        {
            Mat asd = umat.ToMat(AccessType.Read);
            asd.Save(file.Path);
            bitmap = asd.ToWritableBitmap();
            bitmap.Invalidate();

            asd.Dispose();
            asd = null;
                                              
            //Show photo
            imagePreview.Source = bitmap;
            bitmap = null;         

            ScanButton.Visibility = Visibility.Collapsed;
            Back.Visibility = Visibility.Visible;
            Save.Visibility = Visibility.Visible;
            Delete.Visibility = Visibility.Visible;
            toggleOptions.IsEnabled = true;
        }

        private async Task<StorageFile> TakePicture()
        {
            value = ViewType.ViewFinished;
            isRectangle = false;            
            player.Play();            

            //Create JPEG image encoding format
            imageFormat = ImageEncodingProperties.CreateJpeg();

            //Create storage file in local app storage
            StorageFile file = await ApplicationData.Current.LocalFolder.CreateFileAsync("Photo.jpg", CreationCollisionOption.ReplaceExisting);

            //Take photo and store it on file location
            await captureManager.CapturePhotoToStorageFileAsync(imageFormat, file);
            return file;
        }

        protected override void OnNavigatedFrom(NavigationEventArgs e)
        {
            StopCamera();
        }

        private async void Save_Click(object sender, RoutedEventArgs e)
        {
            
            //Create storage folder in Picture Library
            folder = await KnownFolders.PicturesLibrary.CreateFolderAsync("PicScan", CreationCollisionOption.OpenIfExists);
                        
            //Copy photo from file to PicScan album
            await file.CopyAsync(folder, "PicScan.jpg", NameCollisionOption.GenerateUniqueName);           
           
        }

        private async void Back_Click(object sender, RoutedEventArgs e)
        {
            imagePreview.Source = null;
            await captureManager.StartPreviewAsync();

            ScanButton.Visibility = Visibility.Visible;
            Back.Visibility = Visibility.Collapsed;
            Save.Visibility = Visibility.Collapsed;
            Delete.Visibility = Visibility.Collapsed;
            toggleOptions.IsChecked = false;
            toggleOptions.IsEnabled = false;

            DisposeAll();                        
        }

       

        private async void Delete_Click(object sender, RoutedEventArgs e)
        {            
            imagePreview.Source = null;
            await captureManager.StartPreviewAsync();

            ScanButton.Visibility = Visibility.Visible;
            Back.Visibility = Visibility.Collapsed;
            Save.Visibility = Visibility.Collapsed;
            Delete.Visibility = Visibility.Collapsed;
            toggleOptions.IsChecked = false;
            toggleOptions.IsEnabled = false;

            DisposeAll();
        }
        private async void DisposeAll()
        {
            imgNormal.Dispose();
            imgNormal = null;

            imgGaussian.Dispose();
            imgGaussian = null;

            imgRect.Dispose();
            imgRect = null;

            grayUmat.Dispose();
            grayUmat = null;

            pyrDownUmat.Dispose();
            pyrDownUmat = null;

            pyrUpUmat.Dispose();
            pyrUpUmat = null;
                    
            cannyEdges.Dispose();
            cannyEdges = null;

            cannyEdgesDown.Dispose();
            cannyEdgesDown = null;

            contoursUmat.Dispose();
            contoursUmat = null;

            finishedImage.Dispose();
            finishedImage = null;

            await file.DeleteAsync();
        }   

        private void Normal_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewNormal;
            ShowPicture(imgNormal);
        }

        private void Gaussian_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewGaussian;
            ShowPicture(imgGaussian);
        }

        private void Grey_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewGray;
            ShowPicture(grayUmat);
        }
        private void PyrDown_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewPyrDown;
            ShowPicture(pyrDownUmat);
        }

        private void PyrUp_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewPyrUp;
            ShowPicture(pyrUpUmat);
        }

        private void Canny_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewCanny;
            ShowPicture(cannyEdges);
        }

        private void CannyDown_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewCannyDown;
            ShowPicture(cannyEdgesDown);
        }    

        private void Contours_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewContours;
            ShowPicture(contoursUmat);
        }

        private void Rectangle_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewRect;
            ShowPicture(imgRect);
            
        }          

        private void Finished_Checked(object sender, RoutedEventArgs e)
        {
            value = ViewType.ViewFinished;
            if (isRectangle)
                ShowPicture(finishedImage);
            else
                ShowPicture(imgNormal);
        }

        private void toggleOptions_Checked(object sender, RoutedEventArgs e)
        {            
            Normal.Visibility = Visibility.Visible;
            Gaussian.Visibility = Visibility.Visible;
            Grey.Visibility = Visibility.Visible;
            PyrDown.Visibility = Visibility.Visible;
            PyrUp.Visibility = Visibility.Visible;
            Canny.Visibility = Visibility.Visible;
            CannyDown.Visibility = Visibility.Visible;
            Contours.Visibility = Visibility.Visible;
            Rectangle.Visibility = Visibility.Visible;
            Finished.Visibility = Visibility.Visible;
        }

        private void toggleOptions_Unchecked(object sender, RoutedEventArgs e)
        {
            Normal.Visibility = Visibility.Collapsed;
            Gaussian.Visibility = Visibility.Collapsed;
            Grey.Visibility = Visibility.Collapsed;
            PyrDown.Visibility = Visibility.Collapsed;
            PyrUp.Visibility = Visibility.Collapsed;
            Canny.Visibility = Visibility.Collapsed;
            CannyDown.Visibility = Visibility.Collapsed;
            Contours.Visibility = Visibility.Collapsed;
            Rectangle.Visibility = Visibility.Collapsed;
            Finished.Visibility = Visibility.Collapsed;
        }       
                         
        
    }
}
