using System.Runtime.InteropServices;
using System.Windows.Forms;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using WinRT;

namespace RawViewerShare;

internal static class Program
{
    private static readonly Guid DtmIid = new(0xa5caee9b, 0x8708, 0x49d1, 0x8d, 0x36, 0x67, 0xd2, 0x5a, 0x8d, 0xa0, 0x0c);
    private static readonly Dictionary<long, string> PendingPaths = new();
    private static readonly HashSet<long> RegisteredHwnds = new();

    [ComImport]
    [Guid("3A3DCD6C-3EAB-43DC-BCDE-45671CE800C8")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDataTransferManagerInterop
    {
        IntPtr GetForWindow(IntPtr appWindow, ref Guid riid);
        void ShowShareUIForWindow(IntPtr appWindow);
    }

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: WindowsShareHelper <hwnd> <file-path>");
            return 2;
        }

        if (!long.TryParse(args[0], out var hwndValue) || hwndValue == 0)
        {
            return 3;
        }

        var path = Path.GetFullPath(args[1]);
        if (!File.Exists(path))
        {
            return 4;
        }

        try
        {
            if (!ShareFile(hwndValue, path))
            {
                return 1;
            }

            // Keep pumping messages until DataRequested completes or the user dismisses share UI.
            Application.Run(new ApplicationContext());
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static bool ShareFile(long hwndValue, string path)
    {
        PendingPaths[hwndValue] = path;
        var hwnd = new IntPtr(hwndValue);
        var interop = DataTransferManager.As<IDataTransferManagerInterop>();

        if (!RegisteredHwnds.Contains(hwndValue))
        {
            var raw = interop.GetForWindow(hwnd, DtmIid);
            var manager = MarshalInterface<DataTransferManager>.FromAbi(raw);
            var capturedHwnd = hwndValue;
            manager.DataRequested += (_, args) => OnDataRequested(capturedHwnd, args);
            RegisteredHwnds.Add(hwndValue);
        }

        interop.ShowShareUIForWindow(hwnd);
        return true;
    }

    private static async void OnDataRequested(long hwndValue, DataRequestedEventArgs args)
    {
        if (!PendingPaths.TryGetValue(hwndValue, out var path) || string.IsNullOrEmpty(path))
        {
            Application.ExitThread();
            return;
        }

        var deferral = args.Request.GetDeferral();
        try
        {
            var file = await StorageFile.GetFileFromPathAsync(path);
            args.Request.Data.SetStorageItems(new[] { file });
            args.Request.Data.Properties.Title = Path.GetFileName(path);
            args.Request.Data.RequestedOperation = DataPackageOperation.Copy;
        }
        finally
        {
            deferral.Complete();
            Application.ExitThread();
        }
    }
}
