using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using OpenCvSharp;
using TensorFlow;

namespace TensorflowMobileNet
{
    class Program
    {
        public static IEnumerable<CatalogItem> Catalog;
        private static string _input = "input.jpg";
        private static string _output = "output.jpg";
        private static string _catalogPath = "mscoco_label_map.pbtxt";
        private static string _modelPath = "frozen_inference_graph.pb";

        static void Main()
        {
            var img = Cv2.ImRead(_input);

            Catalog = CatalogUtil.ReadCatalogItems(_catalogPath);
            var fileTuples = new List<(string input, string output)> { (_input, _output) };
            string modelFile = _modelPath;

            using (var graph = new TFGraph())
            {
                var model = File.ReadAllBytes(modelFile);
                graph.Import(new TFBuffer(model));
                using (var session = new TFSession(graph))
                {
                    foreach (var tuple in fileTuples)
                    {
                        var tensor = ImageUtil.CreateTensorFromImageFile(tuple.input, TFDataType.UInt8);
                        var runner = session.GetRunner();
                        runner
                            .AddInput(graph["image_tensor"][0], tensor)
                            .Fetch(
                                graph["detection_boxes"][0],
                                graph["detection_scores"][0],
                                graph["detection_classes"][0],
                                graph["num_detections"][0]);

                        Stopwatch sw = new Stopwatch();
                        sw.Start();
                        var output = runner.Run();
                        sw.Stop();
                        Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

                        var boxes = (float[,,])output[0].GetValue(jagged: false);
                        var scores = (float[,])output[1].GetValue(jagged: false);
                        var classes = (float[,])output[2].GetValue(jagged: false);
                        var num = (float[])output[3].GetValue(jagged: false);

                        #region show image

                        for (int i = 0; i < boxes.GetLength(1);i++)
                        {
                            if (scores[0, i] > 0.5)
                            {
                                var idx = Convert.ToInt32(classes[0, i]);
                                var x1 = (int)(boxes[0, i, 1] * img.Width);
                                var y1 = (int)(boxes[0, i, 0] * img.Height);
                                var x2 = (int)(boxes[0, i, 3] * img.Width);
                                var y2 = (int)(boxes[0, i, 2] * img.Height);
                                var catalog = Catalog.First(x => x.Id == idx);
                                string label = $"{(catalog == null ? idx.ToString() : catalog.DisplayName)}: {scores[0, i] * 100:0.00}%";
                                Console.WriteLine($"{label} {x1} {y1} {x2} {y2}");
                                Cv2.Rectangle(img,new Rect(x1, y1, x2-x1, y2-y1),Scalar.Red,2);
                                int baseline;
                                var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out baseline);
                                textSize.Height = textSize.Height + baseline/2;
                                var y = y1 - textSize.Height < 0 ? y1 + textSize.Height : y1;
                                Cv2.Rectangle(img,new Rect(x1,y-textSize.Height,textSize.Width,textSize.Height + baseline / 2),Scalar.Red,Cv2.FILLED);
                                Cv2.PutText(img,label,new Point(x1,y),HersheyFonts.HersheyTriplex, 0.5,Scalar.Black);
                            }
                        }
                        #endregion
                    }
                }
            }
            using (new Window("image", img))
            {
                Cv2.WaitKey();
            }
        }
    }

    public static class CatalogUtil
    {
		private static string CATALOG_ITEM_PATTERN = "item {\n  name: \"(?<name>.*)\"\n  id: (?<id>\\d+)\n  display_name: \"(?<displayName>.*)\"\n}";
        /// <summary>
        /// Reads catalog of well-known objects from text file.
        /// </summary>
        /// <param name="file">path to the text file</param>
        /// <returns>collection of items</returns>
        public static IEnumerable<CatalogItem> ReadCatalogItems(string file)
        {
            using (FileStream stream = File.OpenRead(file))
            using (StreamReader reader = new StreamReader(stream))
            {
                string text = reader.ReadToEnd();
                if (string.IsNullOrWhiteSpace(text))
                {
                    yield break;
                }

                Regex regex = new Regex (CATALOG_ITEM_PATTERN);
                var matches = regex.Matches(text);
                foreach (Match match in matches)
                {
                    var name = match.Groups[1].Value;
                    var id = int.Parse(match.Groups[2].Value);
                    var displayName = match.Groups[3].Value;

                    yield return new CatalogItem
                    {
                        Id = id,
                        Name = name,
                        DisplayName = displayName
                    };
                }
            }
        }
    }

    public class CatalogItem
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string DisplayName { get; set; }
    }

    public static class ImageUtil
    {
        // Convert the image in filename to a Tensor suitable as input to the Inception model.
        public static TFTensor CreateTensorFromImageFile(string file, TFDataType destinationDataType = TFDataType.Float)
        {
            var contents = File.ReadAllBytes(file);

            // DecodeJpeg uses a scalar String-valued tensor as input.
            var tensor = TFTensor.CreateString(contents);

            TFGraph graph;
            TFOutput input, output;

            // Construct a graph to normalize the image
            ConstructGraphToNormalizeImage(out graph, out input, out output, destinationDataType);

            // Execute that graph to normalize this one image
            using (var session = new TFSession(graph))
            {
                var normalized = session.Run(
                    inputs: new[] { input },
                    inputValues: new[] { tensor },
                    outputs: new[] { output });

                return normalized[0];
            }
        }

        // The inception model takes as input the image described by a Tensor in a very
        // specific normalized format (a particular image size, shape of the input tensor,
        // normalized pixel values etc.).
        //
        // This function constructs a graph of TensorFlow operations which takes as
        // input a JPEG-encoded string and returns a tensor suitable as input to the
        // inception model.
        private static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output, TFDataType destinationDataType = TFDataType.Float)
        {
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained after with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.

            const int w = 224;
            const int h = 224;
            const float mean = 117;
            const float scale = 1;

            graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);

            output = graph.Cast(graph.Div(
                x: graph.Sub(
                    x: graph.ResizeBilinear(
                        images: graph.ExpandDims(
                            input: graph.Cast(
                                graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float),
                            dim: graph.Const(0, "make_batch")),
                        size: graph.Const(new int[] { w, h }, "size")),
                    y: graph.Const(mean, "mean")),
                y: graph.Const(scale, "scale")), destinationDataType);
        }
    }

}
