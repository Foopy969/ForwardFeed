using System;
using System.Linq;

namespace ForwardFeed
{
    public class Extensions
    {
        private static double count;
        private static double loss;
        private static double accuracy;
        private static double total;
        private static double progress;

        public static void Log(string text)
        {
            Console.WriteLine($"[{DateTime.Now.ToString("HH:mm:ss")}]{text}");
        }
        public static void Seperate()
        {
            Console.WriteLine("------------------------------------------------------------------");
        }
        public static void ProgressBar()
        {
            int oldProgress = (int)progress;
            progress = (int)(100.0 * count / total);
            if (oldProgress != progress)
            {
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine($"[{DateTime.Now.ToString("HH:mm:ss")}][loading] {count} / {total}, {progress}%");
            }
        }
        public static void ResetScore(int input)
        {
            loss = 0;
            accuracy = 0;
            count = 0;
            total = input;
        }
        public static void AddScore(double[] errors)
        {
            count++;
            loss += errors.Select(x => Math.Abs(x)).Sum();
            accuracy += errors.Select(x => Math.Abs(Math.Round(x))).Sum() == 0 ? 100 : 0;
        }
        public static (double, double) GetScore()
        {
            return (loss / total, accuracy / total);
        }
    }
}
