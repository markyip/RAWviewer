using System.Runtime.InteropServices;
using System.Windows.Forms;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using WinRT;

namespace RawViewerShare;

internal static class NativeMethods
{
    private const int GwlpHwndParent = -8;

    [DllImport("user32.dll", EntryPoint = "SetWindowLong")]
    private static extern IntPtr SetWindowLong32(IntPtr hWnd, int nIndex, IntPtr dwNewLong);

    [DllImport("user32.dll", EntryPoint = "SetWindowLongPtr")]
    private static extern IntPtr SetWindowLong64(IntPtr hWnd, int nIndex, IntPtr dwNewLong);

    [DllImport("user32.dll")]
    internal static extern bool SetForegroundWindow(IntPtr hWnd);

    internal static void SetWindowOwner(IntPtr child, IntPtr owner)
    {
        if (child == IntPtr.Zero || owner == IntPtr.Zero)
        {
            return;
        }

        if (IntPtr.Size == 8)
        {
            SetWindowLong64(child, GwlpHwndParent, owner);
        }
        else
        {
            SetWindowLong32(child, GwlpHwndParent, owner);
        }
    }
}

internal static class Program
{
    private static readonly Guid DtmIid = new(0xa5caee9b, 0x8708, 0x49d1, 0x8d, 0x36, 0x67, 0xd2, 0x5a, 0x8d, 0xa0, 0x0c);
    private static readonly Dictionary<long, string> PendingPaths = new();
    private static readonly HashSet<long> RegisteredHwnds = new();
    private static Form? _hostForm;

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
            Console.Error.WriteLine("usage: WindowsShareHelper <owner-hwnd> <file-path>");
            return 2;
        }

        if (!long.TryParse(args[0], out var ownerHwndValue))
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
            if (!ShareFile(ownerHwndValue, path))
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

    private static IntPtr EnsureHostWindow(IntPtr ownerHwnd)
    {
        if (_hostForm == null || _hostForm.IsDisposed)
        {
            _hostForm = new Form
            {
                ShowInTaskbar = false,
                FormBorderStyle = FormBorderStyle.None,
                StartPosition = FormStartPosition.CenterScreen,
                Width = 1,
                Height = 1,
                Opacity = 0,
            };
        }

        if (!_hostForm.IsHandleCreated)
        {
            _hostForm.CreateControl();
        }

        if (!_hostForm.Visible)
        {
            // Keep a real (but invisible) top-level window alive for WinRT on first use.
            _hostForm.Show();
        }

        var hostHwnd = _hostForm.Handle;
        NativeMethods.SetWindowOwner(hostHwnd, ownerHwnd);
        return hostHwnd;
    }

    private static bool ShareFile(long ownerHwndValue, string path)
    {
        // WinRT share APIs require a window handle owned by the calling process.
        // The Python/Qt HWND is only used as an owner for z-order; DTM calls use
        // a hidden helper-owned form handle in this process.
        var ownerHwnd = ownerHwndValue == 0 ? IntPtr.Zero : new IntPtr(ownerHwndValue);
        var hostHwnd = EnsureHostWindow(ownerHwnd);
        var hostKey = hostHwnd.ToInt64();
        PendingPaths[hostKey] = path;

        if (ownerHwnd != IntPtr.Zero)
        {
            NativeMethods.SetForegroundWindow(ownerHwnd);
        }

        var interop = DataTransferManager.As<IDataTransferManagerInterop>();

        if (!RegisteredHwnds.Contains(hostKey))
        {
            var raw = interop.GetForWindow(hostHwnd, DtmIid);
            var manager = MarshalInterface<DataTransferManager>.FromAbi(raw);
            var capturedKey = hostKey;
            manager.DataRequested += (_, args) => OnDataRequested(capturedKey, args);
            RegisteredHwnds.Add(hostKey);
        }

        Application.DoEvents();
        interop.ShowShareUIForWindow(hostHwnd);
        return true;
    }

    private static async void OnDataRequested(long hostKey, DataRequestedEventArgs args)
    {
        if (!PendingPaths.TryGetValue(hostKey, out var path) || string.IsNullOrEmpty(path))
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
