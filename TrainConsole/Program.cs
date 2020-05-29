using System;
using Microsoft.ML;
using Shared;
using System.Linq;



namespace TrainConsole
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @"C:\Users\runeetv\Downloads\mlnet-workshop-master\data\true_car_listings.csv";
        private static string MODEL_FILEPATH = @"C:\Users\runeetv\Downloads\mlnet-workshop-master\Model\MLModel.zip";



        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            MLContext mlContext = new MLContext();

            // Load training data
            Console.WriteLine("Loading data...");
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');
            Console.WriteLine("Done Loading data...");

            // Split the data into a train and test set
            var trainTestSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);

            // Create data transformation pipeline
            var dataProcessPipeline =
                mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
                    .Append(mlContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
                    .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                    .AppendCacheCheckpoint(mlContext);

            // Choose an algorithm and add to the pipeline
            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model
            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
            Console.WriteLine("Training model done.");

            // Make predictions on train and test sets
            IDataView trainSetPredictions = model.Transform(trainTestSplit.TrainSet);
            IDataView testSetpredictions = model.Transform(trainTestSplit.TestSet);

            // Calculate evaluation metrics for train and test sets
            var trainSetMetrics = mlContext.Regression.Evaluate(trainSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
            var testSetMetrics = mlContext.Regression.Evaluate(testSetpredictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"Train Set R-Squared: {trainSetMetrics.RSquared} | Test Set R-Squared {testSetMetrics.RSquared}");

            var crossValidationResults = mlContext.Regression.CrossValidate(trainingData, trainingPipeline, numberOfFolds: 5);
            var avgRSquared = crossValidationResults.Select(model => model.Metrics.RSquared).Average();
            Console.WriteLine($"Cross Validated R-Squared: {avgRSquared}");

            // Save model
            Console.WriteLine("Saving model...");
            mlContext.Model.Save(model, trainingData.Schema, MODEL_FILEPATH);

        }
    }
}
