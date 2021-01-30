using System;
using System.Collections.Generic;
using System.IO;

namespace ForwardFeed
{
    public static class MnistReader
    {
        private const string TrainImages = "mnist/train-images.idx3-ubyte";
        private const string TrainLabels = "mnist/train-labels.idx1-ubyte";
        private const string TestImages = "mnist/t10k-images.idx3-ubyte";
        private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }
        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }

            labels.Close();
            images.Close();
        }
    }

    public static class MnistExtensions
    {

        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }

        public static string DisplayImage(byte[,] Data)
        {
            string Text = "";
            for (int i = 0; i < Data.GetLength(0); i++)
            {
                for (int j = 0; j < Data.GetLength(1); j++)
                {
                    if (Data[i, j] == 0)
                    {
                        Text += ' ';
                    }
                    else if (Data[i, j] < 64)
                    {
                        Text += '░';
                    }
                    else if (Data[i, j] < 128)
                    {
                        Text += '▒';
                    }
                    else if (Data[i, j] < 196)
                    {
                        Text += '▓';
                    }
                    else
                    {
                        Text += '█';
                    }
                }
                Text += '\n';
            }
            return Text;
        }
        public static T[,] Make2DArray<T>(T[] input, int height, int width)
        {
            T[,] output = new T[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    output[i, j] = input[i * width + j];
                }
            }
            return output;
        }

        public static IList<T> Shuffle<T>(this IList<T> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }

            return list;
        }
    }

    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }
    }
    public class MatrixDeserialze
    {
        public double[] Values { get; set; }
        public int ColumnCount { get; set; }
        public int RowCount { get; set; }
    }
}
