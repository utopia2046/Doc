using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace KeyboardTest
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            KeyEventsHandler("KeyDown", e);
        }

        private void Window_KeyUp(object sender, KeyEventArgs e)
        {
            KeyEventsHandler("KeyUp", e);
            ClearCheckBoxes();
        }

        private void KeyEventsHandler(string eventType, KeyEventArgs e)
        {
            StringBuilder sb = new StringBuilder(eventType);

            txbEventType.Text = eventType;
            sb.Append(": ");

            txbKeyName.Text = e.Key.ToString();
            sb.Append(e.Key.ToString());

            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Control))
            {
                ckbCtrl.IsChecked = true;
                if (e.Key != Key.LeftCtrl && e.Key != Key.RightCtrl)
                {
                    sb.Append(" + Ctrl");
                }
            }

            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
            {
                ckbShift.IsChecked = true;
                if (e.Key != Key.LeftShift && e.Key != Key.RightShift)
                { 
                    sb.Append(" + Shift");
                }
            }

            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Alt))
            {
                ckbAlt.IsChecked = true;
                if (e.Key != Key.LeftAlt && e.Key != Key.RightAlt)
                { 
                    sb.Append(" + Alt");
                }
            }

            lstList.Items.Add(sb.ToString());
            lstList.ScrollIntoView(lstList.Items[lstList.Items.Count - 1]);
        }

        private void ClearCheckBoxes()
        {
            ckbAlt.IsChecked = false;
            ckbCtrl.IsChecked = false;
            ckbShift.IsChecked = false;
        }

        private void btnClear_Click(object sender, RoutedEventArgs e)
        {
            lstList.Items.Clear();
        }
    }
}