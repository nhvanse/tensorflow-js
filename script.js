import { MnistData } from './data.js';

window.data;
window.model;

async function showExamples(data) {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}




function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.

    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));


    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}


async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
    console.log(TRAIN_DATA_SIZE);
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}



async function run() {
    // const data = new MnistData();
    // await data.load();
    let data = window.data;
    // await showExamples(data);
    const model = getModel();
    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

    await train(model, data);
    window.model = model;
    await model.save('downloads://my-model');
}

async function loadSavedModel(path) {
    window.model = await tf.loadLayersModel(path);
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(testDataSize = 500) {
    let model = window.model;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);
    testxs.dispose();
    return [preds, labels];
}

async function showAccuracy() {
    const [preds, labels] = doPrediction();
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {
        name: 'Accuracy',
        tab: 'Evaluation'
    };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
    labels.dispose();
}

async function showConfusion() {
    const [preds, labels] = doPrediction();
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = {
        name: 'Confusion Matrix',
        tab: 'Evaluation'
    };
    tfvis.render.confusionMatrix(container, {
        values: confusionMatrix,
        tickLabels: classNames
    });
    labels.dispose();
}

async function testImage() {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Test', tab: 'Test' });

    // Get the examples
    const examples = data.nextTestBatch(1);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        let model = window.model;
        const testxs = imageTensor.reshape([1, 28, 28, 1]);
        const preds = model.predict(testxs).argMax([-1]);
        setTimeout(function() {
            alert(preds.flatten());
        },2);

        imageTensor.dispose();
    }
}




document.addEventListener('DOMContentLoaded', async e => {
    document.querySelector('#load-data').disabled = false;
    document.querySelector('#load-model').disabled = true;
    document.querySelector('#show-examples').disabled = true;
    document.querySelector('#show-accuracy').disabled = true;
    document.querySelector('#show-confusion').disabled = true;
    document.querySelector('#train').disabled = true;
    document.querySelector('#test').disabled = true;
});

document.querySelector('#load-data').addEventListener('click', async e => {
    window.data = new MnistData();
    await window.data.load();

    e.target.disabled = true;
    document.querySelector('#load-model').disabled = false;
    document.querySelector('#train').disabled = false;
    document.querySelector('#show-examples').disabled = false;

    console.log('load data completely');
    alert('load data completely');
});

document.querySelector('#show-examples').addEventListener('click', async e => {
    showExamples(window.data);
});

document.querySelector('#train').addEventListener('click', async e => {
    document.querySelector('#train').disabled = true;
    document.querySelector('#show-accuracy').disabled = true;
    document.querySelector('#show-confusion').disabled = true;
    await run();
    document.querySelector('#show-accuracy').disabled = false;
    document.querySelector('#show-confusion').disabled = false;
    document.querySelector('#test').disabled = false;
});

document.querySelector('#load-model').addEventListener('click', () => {
    loadSavedModel('./model/big-model.json');
    document.querySelector('#show-accuracy').disabled = false;
    document.querySelector('#show-confusion').disabled = false;
    alert('load model completely.');
    document.querySelector('#test').disabled = false;

});
document.querySelector('#show-accuracy').addEventListener('click', () => showAccuracy());
document.querySelector('#show-confusion').addEventListener('click', () => showConfusion());
document.querySelector('#test').addEventListener('click', () => testImage());
