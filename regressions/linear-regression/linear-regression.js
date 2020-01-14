const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this._processFeatures(features);
    this.labels = tf.tensor(labels);

    this.weights = tf.zeros([this.features.shape[1], 1]);

    this.mseHistory = [];
    // this.bHistory = [];

    this.options = { learningRate: 0.1, iterations: 1000, ...options };

    this.learningRateHistory = [this.options.learningRate];
  }

  _processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = this._standardize(features, this.mean, this.variance);
    } else {
      features = this._standardizeInit(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  _standardizeInit(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return this._standardize(features, mean, variance);
  }

  _standardize(features, mean, variance) {
    return features.sub(mean).div(variance.sqrt());
  }

  _recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();
    this.mseHistory.unshift(mse);
  }

  _recordLearningRate() {
    this.learningRateHistory.unshift(this.options.learningRate);
  }

  _updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }

    this._recordLearningRate();
  }

  _slicer(tensor, batchStartRow) {
    return tensor.slice([batchStartRow, 0], [this.options.batchSize, -1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);

    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    /*
    lodash
      const guessesForMPG = _.map(this.features, row => this.m * row[0] + this.b);

      const mSlope = _.chain(guessesForMPG)
        .map((guess, i) => -1 * this.features[i][0] * (this.labels[i][0] - guess))
        .sum()
        // .multiply(2)
        .divide(this.features.length)
        .value();

      const bSlope = _.chain(guessesForMPG)
        .map((guess, i) => guess - this.labels[i][0])
        .sum()
        // .multiply(2)
        .divide(this.features.length)
        .value();

      this.m -= mSlope * this.options.learningRate;
      this.b -= bSlope * this.options.learningRate;
    */
  }

  train() {
    const { batchSize, iterations } = this.options;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);
    for (let i = 0; i < iterations; i++) {
      // this.bHistory.unshift(this.weights.arraySync()[0][0]);
      for (let j = 0; j < batchQuantity; j++) {
        const batchStartRow = j * batchSize;
        const featureSlice = this._slicer(this.features, batchStartRow);
        const labelSlice = this._slicer(this.labels, batchStartRow);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this._recordMSE();
      this._updateLearningRate();
    }

    /*
    Non-batch gradient descent

    for (let i = 0; i < iterations; i++) {
      // this.bHistory.unshift(this.weights.arraySync()[0][0]);
      this.gradientDescent(this.features, this.labels);
      this._recordMSE();
      this._updateLearningRate();
    }
    */
  }

  test(testFeatures, testLabels) {
    testFeatures = this._processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const ssr = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .arraySync();

    const tss = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .arraySync();

    return 1 - ssr / tss;
  }

  predict(observations) {
    return this._processFeatures(observations).matMul(this.weights);
  }
}

module.exports = LinearRegression;
