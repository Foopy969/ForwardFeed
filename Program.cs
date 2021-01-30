using System;
using System.IO;
using System.Linq;

namespace ForwardFeed
{
    class Program
    {
        static void Main(string[] args)
        {
            //NeuralNetwork Nobject = new NeuralNetwork(784, 16, 16, 10);
            //Nobject.Randomize();
            NeuralNetwork Nobject = new NeuralNetwork(new Model(ReadFromBinaryFile<(double[,], double[])[]>("storage/checkpoint2.txt")));

            Extensions.Log("[info] reading data...");
            Image[] TestData = MnistReader.ReadTestData().ToArray();
            Image[] TrainingData = MnistReader.ReadTrainingData().ToArray();

            (double Loss, double Acc) result;

            for (int i = 0; i < 100; i++)
            {
                Extensions.Seperate();
                Extensions.Log($"[info] epoch {i}");
                Extensions.Log($"[info] training...");
                result = Nobject.Train(MakeData(TrainingData), NeuralNetwork.Optimizer.Nesterov);
                Extensions.Log($"[info] Loss: [{result.Loss.ToString("N5")}], Acc: [{result.Acc.ToString("N3")}%]");
                Extensions.Log($"[info] evaluating...");
                result = Nobject.Evaluate(MakeData(TestData));
                Extensions.Log($"[info] Loss: [{result.Loss.ToString("N5")}], Acc: [{result.Acc.ToString("N3")}%]");

                if (double.IsNaN(result.Loss))
                {
                    Extensions.Log("[error] catastrphic error no recover");
                    Console.ReadKey();
                }
                else 
                    WriteToBinaryFile("storage/checkpoint2.txt", Nobject.CheckPoint.GetCheckPoints());
            }

            Console.ReadKey();
        }

        private static (double[] Input, double[] Answer)[] MakeData(Image[] Images)
        {
            Extensions.Log("[info] shuffling data...");
            return Images.Shuffle().Select(e => (e.Data.Cast<byte>().Select(x => x / 255.0)
            .ToArray(), new double[10].Select((x, y) => Convert.ToDouble(y == e.Label)).ToArray())).ToArray();
        }
        public static void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
        {
            using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }
        public static T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                return (T)binaryFormatter.Deserialize(stream);
            }
        }
    }
}
