const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this._processFeatures(features);
    this.labels = tf.tensor(labels);

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);

    this.costHistory = [];
    // this.bHistory = [];

    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      decisionBoundary: 0.5,
      ...options,
    };

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
    const filler = variance
      .cast('bool')
      .logicalNot()
      .cast('float32');
    const varianceWithFiller = variance.add(filler);
    return features.sub(mean).div(varianceWithFiller.sqrt());
  }

  _recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).softmax();

      const term1 = this.labels.transpose().matMul(
        guesses
          .add(1e-7) // 1 * 10^-7 avoids log(0)
          .log()
      );

      const term2 = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(
          guesses
            .mul(-1)
            .add(1)
            .add(1e-7) // 1 * 10^-7 avoids log(0)
            .log()
        );

      return term1
        .add(term2)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync()[0][0];
    });

    this.costHistory.unshift(cost);
  }

  _recordLearningRate() {
    this.learningRateHistory.unshift(this.options.learningRate);
  }

  _updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }
    if (this.costHistory[0] > this.costHistory[1]) {
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
    const currentGuesses = features.matMul(this.weights).softmax();

    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
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

        this.weights = tf.tidy(() => {
          const featureSlice = this._slicer(this.features, batchStartRow);
          const labelSlice = this._slicer(this.labels, batchStartRow);
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this._recordCost();
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
    this.predictions = this.predict(testFeatures);
    this.testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = this.predictions
      .notEqual(this.testLabels)
      .sum()
      .arraySync();

    return (this.predictions.shape[0] - incorrect) / this.predictions.shape[0];
  }

  predict(observations) {
    return this._processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }
}

module.exports = LogisticRegression;
