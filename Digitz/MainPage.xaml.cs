using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Graphics.Display;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace Digitz
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
            inky.InkPresenter.InputDeviceTypes =
                Windows.UI.Core.CoreInputDeviceTypes.Mouse |
                Windows.UI.Core.CoreInputDeviceTypes.Pen;
        }

        private async Task<float[]> GetWrittenDigit()
        {
            RenderTargetBitmap b = new RenderTargetBitmap();
            await b.RenderAsync(inky);
            var pixelBuffer = await b.GetPixelsAsync();
            // formats correctly....?
            SoftwareBitmap bitmap = SoftwareBitmap.CreateCopyFromBuffer(pixelBuffer, BitmapPixelFormat.Gray8, 28, 28);
            var pixels = pixelBuffer.ToArray();
            float[] grayscale = new float[28 * 28];
            for (int i = 0; i < pixels.Length; i += 4)
                grayscale[i / 4] = 255 - pixels[i + 3];
            return grayscale;
        }


        private async void CameraButton_Click(object sender, RoutedEventArgs e)
        {
            var pixels = await GetWrittenDigit();
        }
    }
}
