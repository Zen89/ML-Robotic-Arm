using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

// definicja klasy danych wejściowych
public class InputData
{
    [LoadColumn(0)] public float Servo1Angle { get; set; }
    [LoadColumn(1)] public float Servo2Angle { get; set; }
    [LoadColumn(2)] public float Servo1Current { get; set; }
    [LoadColumn(3)] public float Servo2Current { get; set; }
    [LoadColumn(4)] public float Servo1PWM { get; set; }
    [LoadColumn(5)] public float Servo2PWM { get; set; }
    [LoadColumn(6)] public float AccX { get; set; }
    [LoadColumn(7)] public float AccY { get; set; }
    [LoadColumn(8)] public float AccZ { get; set; }
    [LoadColumn(9)] public float GyroX { get; set; }
    [LoadColumn(10)] public float GyroY { get; set; }
    [LoadColumn(11)] public float GyroZ { get; set; }
    [LoadColumn(12)] public float MagX { get; set; }
    [LoadColumn(13)] public float MagY { get; set; }
    [LoadColumn(14)] public float MagZ { get; set; }
    [LoadColumn(15)] public float TargetX { get; set; }
    [LoadColumn(16)] public float TargetY { get; set; }
    [LoadColumn(17)] public float TargetZ { get; set; }
}

// definicja klasy danych wyjściowych
public class OutputData
{
    public float CoordinateX { get; set; }
    public float CoordinateY { get; set; }
    public float CoordinateZ { get; set; }

    [ColumnName(nameof(Label)), LoadColumn(0)]
    public float Label { get; set; }

    [ColumnName("PredictedLabel")]
    public uint PredictedLabel { get; set; }

    [LoadColumn(1)]
    public float CurrentIntensityMotor1 { get; set; }

    [LoadColumn(2)]
    public float CurrentIntensityMotor2 { get; set; }

    [LoadColumn(3)]
    public float PWM1 { get; set; }

    [LoadColumn(4)]
    public float PWM2 { get; set; }

    [LoadColumn(5)]
    public float AccelerometerX { get; set; }

    [LoadColumn(6)]
    public float AccelerometerY { get; set; }

    [LoadColumn(7)]
    public float AccelerometerZ { get; set; }

    [LoadColumn(8)]
    public float GyroscopeX { get; set; }

    [LoadColumn(9)]
    public float GyroscopeY { get; set; }

    [LoadColumn(10)]
    public float GyroscopeZ { get; set; }

    [LoadColumn(11)]
    public float MagnetometerX { get; set; }

    [LoadColumn(12)]
    public float MagnetometerY { get; set; }

    [LoadColumn(13)]
    public float MagnetometerZ { get; set; }
}



class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // wczytanie danych treningowych z pliku CSV
        var data = mlContext.Data.LoadFromTextFile<InputData>("training_data.csv", separatorChar: ',');

        // podział danych na zbiory treningowy i testowy
        var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.3);

        // definicja konwertera danych wejściowych
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(OutputData.Label))
            .Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.Servo1Angle), nameof(InputData.Servo1Current), nameof(InputData.Servo1PWM), nameof(InputData.Servo2Angle), nameof(InputData.Servo2Current), nameof(InputData.Servo2PWM), nameof(InputData.AccX), nameof(InputData.AccY), nameof(InputData.AccZ), nameof(InputData.GyroX), nameof(InputData.GyroY), nameof(InputData.GyroZ), nameof(InputData.MagX), nameof(InputData.MagY), nameof(InputData.MagZ)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Transforms.Concatenate("FeaturesNormalized", "Features", nameof(OutputData.Label)))
            .Append(mlContext.Transforms.DropColumns(nameof(OutputData.CoordinateX), nameof(OutputData.CoordinateY), nameof(OutputData.CoordinateZ)))
            .Append(mlContext.Transforms.DropColumns(nameof(InputData.AccX), nameof(InputData.AccY), nameof(InputData.AccZ), nameof(InputData.GyroX), nameof(InputData.GyroY), nameof(InputData.GyroZ), nameof(InputData.MagX), nameof(InputData.MagY), nameof(InputData.MagZ)))
            .Append(mlContext.Transforms.CopyColumns("Label", nameof(OutputData.Label)))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


        // wybór algorytmu uczenia maszynowego i konfiguracja
        var trainer = mlContext.Regression.Trainers.Sdca(); //.FastTree();
        var trainingPipeline = pipeline.Append(trainer);

        // trening modelu
        var trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

        // ewaluacja modelu na zbiorze testowym
        var predictions = trainedModel.Transform(dataSplit.TestSet);
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        // wyświetlenie wyników ewaluacji
        Console.WriteLine($"R2 score: {metrics.RSquared}");
        Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"MSE: {metrics.MeanSquaredError}");

        // zapisanie modelu do pliku
        using (var stream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Model.Save(trainedModel, data.Schema, stream);
        }

        // wizualizacja danych
        var preview = data.Preview();
        Console.WriteLine($"Number of rows: {preview.RowView.Length}");
        Console.WriteLine(string.Join(", ", preview.ColumnView.Select(c => c.Column.Name)));

        // wizualizacja wyników predykcji
        var predictionsPreview = mlContext.Data.CreateEnumerable<OutputData>(predictions, reuseRowObject: false).Take(10);
        foreach (var prediction in predictionsPreview)
        {
            Console.WriteLine($"Predicted coordinates: ({prediction.CoordinateX}, {prediction.CoordinateY}, {prediction.CoordinateZ})");
        }


    }
}
