
using CarbonCapture.DAL.Models.Compliance.HeavyOlefinsTanks;
using ExcelDataReader;
using CarbonCapture.DAL.DataAccess.Compliance.HeavyOlefinsTanks;
using CarbonCapture.BLL.Compliance.HeavyOlefinsTanks;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Preprocessing
{
    public class Preprocessing
    {
        public static void SaveAsJson<T>(List<T> data, string filePath)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true, // Pretty print JSON
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull, // Ignore null values
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals // Handle NaN, Infinity
            };
            string json = JsonSerializer.Serialize(data, options);
            File.WriteAllText(filePath, json);
        }

        private static (double Value, string TentativeQuality) ParseDoubleWithQuality(object obj, string status)
        {
            try
            {
                // Attempt to parse the value as a double
                double value = double.Parse(obj.ToString());
                string quality = DetermineTentativeQuality(status); // Quality based on status
                return (value, quality);
            }
            catch
            {
                // If parsing fails, return NaN and quality as "B"
                return (double.NaN, "B");
            }
        }
        private static string DetermineTentativeQuality(string status)
        {
            return status switch
            {
                "OOS" => "B",
                "IN SERVICE" => "G",
                _ => "B", // Default to "B" if status is unknown
            };
        }

        public static List<List<TemperatureDataModel>> LoadRawTempData(List<AnnualLimitsAndRecordsModel> historicalData)
        {
            // List<List<TemperatureDataModel>>
            // return a list of all months of temperature data model
            // each month corresponds to a list of temperature data model
            // Wrapper list to hold lists of TemperatureDataModel for each month
            List<List<TemperatureDataModel>> monthlyData = new List<List<TemperatureDataModel>>();
            List<TemperatureDataModel> currentMonthData = new List<TemperatureDataModel>();
            DateTime? currentMonth = null;

            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "../Compliance/HeavyOlefinsTanks.xlsx");
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
            using (var stream = File.Open(filePath, FileMode.Open, FileAccess.Read))
            {
                using (var reader = ExcelDataReader.ExcelReaderFactory.CreateReader(stream))
                {
                    var result = reader.AsDataSet(new ExcelDataSetConfiguration
                    {
                        ConfigureDataTable = _ => new ExcelDataTableConfiguration
                        {
                            UseHeaderRow = true,
                        }
                    });
                    var dataTable = result.Tables["Temp Data"];

                    for (int i = 8; i < dataTable.Rows.Count; i++)
                    {
                        // if (i == 8) // first row
                        // {
                        //     Console.WriteLine($"Timestep : {dataTable.Rows[i][0]}, TBD910: {dataTable.Rows[i][1]}, TBD911: {dataTable.Rows[i][3]}, TBD912: {dataTable.Rows[i][5]}");
                        // }
                        if (string.IsNullOrWhiteSpace(dataTable.Rows[i][0]?.ToString()))
                        {
                            continue;
                        }

                        var row = dataTable.Rows[i];
                        var timeStep = DateTime.Parse(row[0].ToString());

                        // If this row belongs to a new month, finalize the current month and start a new list
                        if (currentMonth == null || timeStep.Month != currentMonth.Value.Month || timeStep.Year != currentMonth.Value.Year)
                        {
                            if (currentMonthData.Count > 0)
                            {
                                monthlyData.Add(currentMonthData); // Add the completed month's data
                            }
                            currentMonthData = new List<TemperatureDataModel>(); // Start a new month's data
                            currentMonth = new DateTime(timeStep.Year, timeStep.Month, 1); // Set the current month
                        }

                        // Find the corresponding month and status
                        var matchingRecord = historicalData.FirstOrDefault(record =>
                            timeStep >= record.Start && timeStep <= record.End);

                        string tbd910Status = matchingRecord?.Status.TBD910 ?? "OOS";
                        var tbd910Tup = ParseDoubleWithQuality(row[1], tbd910Status);

                        string tbd911Status = matchingRecord?.Status.TBD911 ?? "OOS";
                        var tbd911Tup = ParseDoubleWithQuality(row[3], tbd911Status);

                        string tbd912Status = matchingRecord?.Status.TBD912 ?? "OOS";
                        var tbd912Tup = ParseDoubleWithQuality(row[5], tbd912Status);

                        currentMonthData.Add(new TemperatureDataModel
                        {
                            TimeStep = timeStep,
                            TBD910 = new TankMetaData(tbd910Tup.Value, tbd910Tup.TentativeQuality, false),
                            TBD911 = new TankMetaData(tbd911Tup.Value, tbd911Tup.TentativeQuality, false),
                            TBD912 = new TankMetaData(tbd912Tup.Value, tbd912Tup.TentativeQuality, false),
                        });
                    }
                }
            }
            // Add the last month's data if not empty
            if (currentMonthData.Count > 0)
            {
                monthlyData.Add(currentMonthData);
            }
            return monthlyData;
        }

        public static List<List<ThroughputDataModel>> LoadRawThroughputData(List<AnnualLimitsAndRecordsModel> historicalData)
        {
            // List<List<TemperatureDataModel>>
            // return a list of all months of temperature data model
            // each month corresponds to a list of temperature data model
            // Wrapper list to hold lists of TemperatureDataModel for each month
            List<List<ThroughputDataModel>> monthlyData = new List<List<ThroughputDataModel>>();
            List<ThroughputDataModel> currentMonthData = new List<ThroughputDataModel>();
            DateTime? currentMonth = null;

            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "../Compliance/HeavyOlefinsTanks.xlsx");
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
            using (var stream = File.Open(filePath, FileMode.Open, FileAccess.Read))
            {
                using (var reader = ExcelDataReader.ExcelReaderFactory.CreateReader(stream))
                {
                    var result = reader.AsDataSet(new ExcelDataSetConfiguration
                    {
                        ConfigureDataTable = _ => new ExcelDataTableConfiguration
                        {
                            UseHeaderRow = true,
                        }
                    });
                    var dataTable = result.Tables["Throughput Data"];

                    for (int i = 8; i < dataTable.Rows.Count; i++)
                    {
                        // if (i == 8) // first row
                        // {
                        //     Console.WriteLine($"Timestep : {dataTable.Rows[i][1]}, TBD910: {dataTable.Rows[i][2]}, TBD911: {dataTable.Rows[i][4]}, TBD912: {dataTable.Rows[i][6]}, TBD913: {dataTable.Rows[i][8]}, TB3301: {dataTable.Rows[i][10]}, TBD301: {dataTable.Rows[i][12]}, TUT604: {dataTable.Rows[i][14]}, TUT605: {dataTable.Rows[i][16]}, TUT918: {dataTable.Rows[i][18]}, TOL400: {dataTable.Rows[i][20]}, TOL600: {dataTable.Rows[i][22]}, G354: {dataTable.Rows[i][24]}, G356: {dataTable.Rows[i][26]}");
                        //     break;
                        // }
                        if (string.IsNullOrWhiteSpace(dataTable.Rows[i][0]?.ToString()))
                        {
                            continue;
                        }

                        var row = dataTable.Rows[i];
                        var timeStep = DateTime.Parse(row[1].ToString());

                        // If this row belongs to a new month, finalize the current month and start a new list
                        if (currentMonth == null || timeStep.Month != currentMonth.Value.Month || timeStep.Year != currentMonth.Value.Year)
                        {
                            if (currentMonthData.Count > 0)
                            {
                                monthlyData.Add(currentMonthData); // Add the completed month's data
                            }
                            currentMonthData = new List<ThroughputDataModel>(); // Start a new month's data
                            currentMonth = new DateTime(timeStep.Year, timeStep.Month, 1); // Set the current month
                        }

                        // Find the corresponding month and status
                        var matchingRecord = historicalData.FirstOrDefault(record =>
                            timeStep >= record.Start && timeStep <= record.End);

                        var tbd910Status = matchingRecord?.Status.TBD910 ?? "OOS";
                        var tbd910Tup = ParseDoubleWithQuality(row[2], tbd910Status);

                        var tbd911Status = matchingRecord?.Status.TBD911 ?? "OOS";
                        var tbd911Tup = ParseDoubleWithQuality(row[4], tbd911Status);

                        var tbd912Status = matchingRecord?.Status.TBD912 ?? "OOS";
                        var tbd912Tup = ParseDoubleWithQuality(row[6], tbd912Status);

                        var tbd913Status = matchingRecord?.Status.TBD913 ?? "OOS";
                        var tbd913Tup = ParseDoubleWithQuality(row[8], tbd913Status);

                        var tb3301Status = matchingRecord?.Status.TB3301 ?? "OOS";
                        var tb3301Tup = ParseDoubleWithQuality(row[10], tb3301Status);

                        var tbd301Status = matchingRecord?.Status.TBD301 ?? "OOS";
                        var tbd301Tup = ParseDoubleWithQuality(row[12], tbd301Status);

                        var tut604Status = matchingRecord?.Status.TUT604 ?? "OOS";
                        var tut604Tup = ParseDoubleWithQuality(row[14], tut604Status);

                        var tut605Status = matchingRecord?.Status.TUT605 ?? "OOS";
                        var tut605Tup = ParseDoubleWithQuality(row[16], tut605Status);

                        var tut918Status = matchingRecord?.Status.TUT918 ?? "OOS";
                        var tut918Tup = ParseDoubleWithQuality(row[18], tut918Status);

                        var tol400Status = matchingRecord?.Status.TOL400 ?? "OOS";
                        var tol400Tup = ParseDoubleWithQuality(row[20], tol400Status);

                        var tol600Status = matchingRecord?.Status.TOL600 ?? "OOS";
                        var tol600Tup = ParseDoubleWithQuality(row[22], tol600Status);

                        var g354Status = matchingRecord?.Status.G354 ?? "OOS";
                        var g354Tup = ParseDoubleWithQuality(row[24], g354Status);

                        var g356Status = matchingRecord?.Status.G356 ?? "OOS";
                        var g356Tup = ParseDoubleWithQuality(row[26], g356Status);

                        currentMonthData.Add(new ThroughputDataModel
                        {
                            TimeStep = timeStep,
                            TBD910 = new TankMetaData(tbd910Tup.Value, tbd910Tup.TentativeQuality, false),
                            TBD911 = new TankMetaData(tbd911Tup.Value, tbd911Tup.TentativeQuality, false),
                            TBD912 = new TankMetaData(tbd912Tup.Value, tbd912Tup.TentativeQuality, false),
                            TBD913 = new TankMetaData(tbd913Tup.Value, tbd913Tup.TentativeQuality, false),
                            TB3301 = new TankMetaData(tb3301Tup.Value, tb3301Tup.TentativeQuality, false),
                            TBD301 = new TankMetaData(tbd301Tup.Value, tbd301Tup.TentativeQuality, false),
                            TUT604 = new TankMetaData(tut604Tup.Value, tut604Tup.TentativeQuality, false),
                            TUT605 = new TankMetaData(tut605Tup.Value, tut605Tup.TentativeQuality, false),
                            TUT918 = new TankMetaData(tut918Tup.Value, tut918Tup.TentativeQuality, false),
                            TOL400 = new TankMetaData(tol400Tup.Value, tol400Tup.TentativeQuality, false),
                            TOL600 = new TankMetaData(tol600Tup.Value, tol600Tup.TentativeQuality, false),
                            G354 = new TankMetaData(g354Tup.Value, g354Tup.TentativeQuality, false),
                            G356 = new TankMetaData(g356Tup.Value, g356Tup.TentativeQuality, false),
                        });
                    }
                }
            }

            // Add the last month's data if not empty
            if (currentMonthData.Count > 0)
            {
                monthlyData.Add(currentMonthData);
            }

            return monthlyData;
        }



        public static void Main(string [] args)
        {
            // this is how we can reference each tank's status for each month. 
            AnnualLimitsAndRecordsDA annualRecordsDA = new AnnualLimitsAndRecordsDA();
            List<AnnualLimitsAndRecordsModel> historicalData = annualRecordsDA.GetAnnualLimitsAndRecords();
            Console.WriteLine($"{historicalData[0].Start}, {historicalData[0].End}, {historicalData[0].Status.TBD910}");
            Console.WriteLine($"{historicalData[historicalData.Count - 1].Start}, {historicalData[historicalData.Count - 1].End}, {historicalData[historicalData.Count - 1].Status.TBD910}");


            //string dataPage = "Throughput Data";
            Console.WriteLine("testing temperature load");

            List<List<TemperatureDataModel>> tempData = LoadRawTempData(historicalData);
            Console.WriteLine("Testing throughput load");

            List<List<ThroughputDataModel>> throughData = LoadRawThroughputData(historicalData);
            
            List<TemperatureCalculations> tempOutputs = new List<TemperatureCalculations>();
            List<ThroughputCalculations> throughputOutputs = new List<ThroughputCalculations>();

            for (int i = 0; i < tempData.Count; i++)
            {
                List<TemperatureDataModel> monthTemp = tempData[i];
                TemperatureCalculations temperatures = new TemperatureCalculations(monthTemp);
                tempOutputs.Add(temperatures);
            }

            for (int i = 0; i < throughData.Count; i++)
            {
                List<ThroughputDataModel> monthThroughput = throughData[i];
                
                ThroughputCalculations throughputs = new ThroughputCalculations(monthThroughput);
                Console.WriteLine(throughputs.OriginalData[0].TimeStep.ToString());
                throughputOutputs.Add(throughputs);
            }
            Console.WriteLine(tempData.Count);
            Console.WriteLine(throughData.Count);
            Console.WriteLine(throughputOutputs.Count);
           // Console.WriteLine($"Test: {throughputOutputs[0]}");//.OriginalData[0].TimeStep.ToString()}");
            // Now save the tempOutputs and throughputOutputs as two json or csv files. 
            // Save tempOutputs and throughputOutputs as JSON
            string tempOutputsPath = Path.Combine(Directory.GetCurrentDirectory(), "TempOutputs.json");
            string throughputOutputsPath = Path.Combine(Directory.GetCurrentDirectory(), "ThroughputOutputs.json");

            SaveAsJson(tempOutputs, tempOutputsPath);
            SaveAsJson(throughputOutputs, throughputOutputsPath);

            Console.WriteLine($"Saved tempOutputs to {tempOutputsPath}");
            Console.WriteLine($"Saved throughputOutputs to {throughputOutputsPath}");
        }
    }
}