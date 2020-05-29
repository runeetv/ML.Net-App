// This file was auto-generated by ML.NET Model Builder. 

using System;
using MLNETWorkshopML.Model;

namespace MLNETWorkshopML.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Product_name = @"Alisha Solid Women's Cycling Shorts",
                Retail_price = 999F,
                Description = @"Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy Red NavySpecifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts",
                Brand = @"Alisha",
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Category with predicted Category from sample data...\n\n");
            Console.WriteLine($"Product_name: {sampleData.Product_name}");
            Console.WriteLine($"Retail_price: {sampleData.Retail_price}");
            Console.WriteLine($"Description: {sampleData.Description}");
            Console.WriteLine($"Brand: {sampleData.Brand}");
            Console.WriteLine($"\n\nPredicted Category value {predictionResult.Prediction} \nPredicted Category scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
