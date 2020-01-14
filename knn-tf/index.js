require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

const knn = (features, labels, testFeature, k) => {
  const { mean, variance } = tf.moments(features, 0);

  const scaledTestFeature = testFeature.sub(mean).div(variance.sqrt());

  const scaledFeatures = features.sub(mean).div(variance.sqrt());

  const dist = scaledFeatures
    .sub(scaledTestFeature)
    .pow(2)
    .sum(1)
    .sqrt();

  const distLabels = dist.expandDims(1).concat(labels, 1);

  const distLabelsSorted = distLabels
    .unstack()
    .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1));

  const distLabelsSortedSliced = distLabelsSorted.slice(0, k);

  const predictedLabel =
    distLabelsSortedSliced.reduce((acc, pair) => {
      return acc + pair.arraySync()[1];
    }, 0) / k;

  return predictedLabel;
};

let { features, labels, testFeatures, testLabels } = loadCSV(
  './kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price'],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testFeature, i) => {
  const result = knn(features, labels, tf.tensor(testFeature), 10);
  const err = ((testLabels[i][0] - result) / testLabels[i][0]) * 100.0;

  console.log(
    'Test features:',
    testFeature,
    'Guess:',
    result,
    'Actual:',
    testLabels[i][0],
    'Error:',
    err
  );
});

/*
  const k = 2;

  const features = tf.tensor([
    [-121, 47],
    [-121.2, 46.5],
    [-122, 46.4],
    [-120.9, 46.7],
  ]);

  const labels = tf.tensor([[200], [250], [215], [240]]);

  const predictionPoint = tf.tensor([-121, 47]);



  console.log(predictedLabel);
*/
