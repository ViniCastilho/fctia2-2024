let trainFile = document.querySelector('#train-file');
let trainLoad = document.querySelector('#train-load');

let confStatus = document.querySelector('#conf-status');
let confHidden = document.querySelector('#conf-hidden');
let confFunc = document.querySelector('#conf-func');
let confStop = document.querySelector('#conf-stop');
let confErr = document.querySelector('#conf-err');
let confItr = document.querySelector('#conf-itr');
let confBegin = document.querySelector('#conf-begin');

let testStatus = document.querySelector('#test-status');
let testFile = document.querySelector('#test-file');
let testBegin = document.querySelector('#test-begin');

let endTable = document.querySelector('#end-table');

const MAX_HIDDEN = 20;

let training = [];
let knownClasses = [];

const MAX_ERROR = 0.2;
const LEARNING_RATE = 0.1;

let INPUT_NEURONS = 0;
let HIDDEN_NEURONS = 0;
let OUTPUT_NEURONS = 0;

let NETWORK_LAYER_SIZE = [];
let LINK_COUNT = 0;

let weightLayers = [];
let weightedSumLayers = [];
let errorLayers = [];

let CURR_TRANSFER = null;
let CURR_TRANSFER_DERIVED = null;

let trainingMethod = {};
let transferCall = {
    'log': (x) => {
        return 1/(1+Math.exp(-x));
    },
    'dlog': (x) => {
        return transferCall.log(x) * (1 - transferCall.log(x));
    },
    'hyp': (x) => {
        return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x));
    },
    'dhyp': (x) => {
        return 1-(transferCall.hyp(x)**2)
    },
};

function geometricMean(amount) {
    if (amount === 0) return 0;
    let val = 1;
    for (let i = 0; i < amount; i++) {
        val *= (i+1);
    }
    return val ** (1/amount);
}

function trainingLoad(event) {
    let rows = event.target.result.split('\r\n');
    if (rows.length <= 1) rows = event.target.result.split('\n');

    training = [];
    knownClasses = [];
    INPUT_NEURONS = rows[0].split(',').length-1;

    for (let i = 1; i < rows.length; i++) {
        let cols = rows[i].split(',');
        training.push([]);
        for (let j = 0; j < cols.length; j++) {
            let val = parseInt(cols[j]);
            if (isNaN(val)) {
                training.pop();
                break;
            } else {
                training[i-1].push(val);
            }

        }
        let cls = parseInt(cols[cols.length-1]);
        if (!knownClasses.includes(cls) && !isNaN(cls)) {
            knownClasses.push(cls);
        }
    }

    OUTPUT_NEURONS = knownClasses.length;
    let mean = Math.ceil(geometricMean(OUTPUT_NEURONS));
    confHidden.value = mean;
    confHidden.max = MAX_HIDDEN;
    if (MAX_HIDDEN < mean) {
        confHidden.max = mean;
    }
    confStatus.innerHTML = '(TREINAMENTO CARREGADO)';
    console.log(`SAÍDA: (${knownClasses.length}) ${knownClasses};`);
    console.log(`MÉDIA: ${confHidden.value}`);
}

function trainingError(event) { alert('Um erro ocorreu ao ler o arquivo de treinamento.'); }

trainLoad.onclick = () => {
    let upload = trainFile.files[0];
    let reader = new FileReader();
    reader.readAsText(upload, 'UTF-8');
    reader.onload = trainingLoad;
    reader.onerror = trainingError;
}

function forwardPropagation(input) {
    let activations = [input];
    for (let layer = 0; layer < LINK_COUNT; layer++) {
        let layerActivation = [];
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            let weightedSum = 0;
            for (let p = 0; p < NETWORK_LAYER_SIZE[layer]; p++) {
                weightedSum += weightLayers[layer][n][p] * activations[layer][p];
            }
            weightedSumLayers[layer][n] = weightedSum;
            layerActivation.push(CURR_TRANSFER(weightedSum));
        }
        activations.push(layerActivation);
    }
    return activations;
}

function backPropagation(activations, expectedOutput) {
    let outputLayerIndex = LINK_COUNT - 1;
    for (let n = 0; n < NETWORK_LAYER_SIZE[outputLayerIndex + 1]; n++) {
        let output = activations[outputLayerIndex + 1][n];
        let expected = (n + 1 === expectedOutput) ? 1 : 0;
        errorLayers[outputLayerIndex][n] = (output - expected) * CURR_TRANSFER_DERIVED(weightedSumLayers[outputLayerIndex][n]);
    }

    for (let layer = outputLayerIndex - 1; layer >= 0; layer--) {
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            let errorSum = 0;
            for (let nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZE[layer + 2]; nextNeuron++) {
                errorSum += weightLayers[layer + 1][nextNeuron][n] * errorLayers[layer + 1][nextNeuron];
            }
            errorLayers[layer][n] = errorSum * CURR_TRANSFER_DERIVED(weightedSumLayers[layer][n]);
        }
    }
}

function updateWeights(activations, LEARNING_RATE) {
    for (let layer = 0; layer < LINK_COUNT; layer++) {
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            for (let p = 0; p < NETWORK_LAYER_SIZE[layer]; p++) {
                weightLayers[layer][n][p] -= LEARNING_RATE * errorLayers[layer][n] * activations[layer][p];
            }
        }
    }
}

trainingMethod.itr = () => {
    let name = confFunc.options[confFunc.selectedIndex].value;
    CURR_TRANSFER = transferCall[name];
    CURR_TRANSFER_DERIVED = transferCall[`d${name}`];
    let count = parseInt(confItr.value);
    for (let epoch = 0; epoch < count; epoch++) {
        for (let data of training) {
            let input = data.slice(0, INPUT_NEURONS);
            let expectedOutput = data[INPUT_NEURONS];
            let activations = forwardPropagation(input); 
            backPropagation(activations, expectedOutput);
            updateWeights(activations, LEARNING_RATE);
        }
    }
}

confBegin.onclick = () => {
    weightLayers = [];
    weightedSumLayers = [];
    errorLayers = [];

    HIDDEN_NEURONS = parseInt(confHidden.value);

    NETWORK_LAYER_SIZE = [INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS];
    LINK_COUNT = NETWORK_LAYER_SIZE.length-1;

    for (let layer = 0; layer < LINK_COUNT; layer++) {
        weightLayers.push([]);
        weightedSumLayers.push([]);
        errorLayers.push([]);
        const PREVIOUS_LAYER_COUNT = NETWORK_LAYER_SIZE[layer];
        const NEXT_LAYER_COUNT = NETWORK_LAYER_SIZE[layer+1];
        for (let n = 0; n < NEXT_LAYER_COUNT; n++) {
            weightedSumLayers[layer].push(0);
            errorLayers[layer].push(0);
            weightLayers[layer].push([]);
            for (let p = 0; p < PREVIOUS_LAYER_COUNT; p++) {
                weightLayers[layer][n].push(Math.random());
            }
        }
    }

    // BEGIN TRAINING
    let opt = confStop.options[confStop.selectedIndex].value;
    trainingMethod[opt]();
    testStatus.innerHTML = '(REDE NEURAL PRONTA)';
}

function testError(event) { alert('Um erro ocorreu ao ler o arquivo de treinamento.'); }

let testing = [];

function testNetwork(input, expectedOutput) {
    let activations = [input];
    for (let layer = 0; layer < LINK_COUNT; layer++) {
        let layerActivation = [];
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            let weightedSum = 0;
            for (let p = 0; p < NETWORK_LAYER_SIZE[layer]; p++) {
                weightedSum += weightLayers[layer][n][p] * activations[layer][p];
            }
            layerActivation.push(CURR_TRANSFER(weightedSum));
        }
        activations.push(layerActivation);
    }

    let outputLayer = activations[LINK_COUNT];

    let predictedClass = outputLayer.indexOf(Math.max(...outputLayer)) + 1;
    console.log(input);
    console.log(`EXPECTED ${expectedOutput}, GOT ${predictedClass}`);
    console.log(outputLayer[predictedClass-1]);
    console.log(outputLayer);
    console.log('\n\n');
    return predictedClass;
}

function testLoad(event) {
    let rows = event.target.result.split('\r\n');
    if (rows.length <= 1) rows = event.target.result.split('\n');

    testing = [];

    for (let i = 1; i < rows.length; i++) {
        let cols = rows[i].split(',');
        testing.push([]);
        for (let j = 0; j < cols.length; j++) {
            let val = parseInt(cols[j]);
            if (isNaN(val)) {
                testing.pop();
                break;
            } else {
                testing[i-1].push(val);
            }

        }
    }

    console.log(testing);
    let mat = [];
    endTable.innerHTML = '';
    for (let i = 0; i < OUTPUT_NEURONS; i++) {
        mat.push([])
        let tr = document.createElement('tr');
        for (let j = 0; j < OUTPUT_NEURONS; j++) {
            let td = document.createElement('td');
            td.innerHTML = '0';
            tr.appendChild(td);
            mat[i].push(0);
        }
        endTable.appendChild(tr);
    }

    for (let data of testing) {
        let input = data.slice(0, INPUT_NEURONS);
        let expectedOutput = data[INPUT_NEURONS];
        let output = testNetwork(input, expectedOutput);
        mat[output-1][expectedOutput-1] = mat[output-1][expectedOutput-1] + 1;
    }
    for (let i = 0; i < OUTPUT_NEURONS; i++) {
        let tr = endTable.children[i];
        for (let j = 0; j < OUTPUT_NEURONS; j++) {
            tr.children[j].innerHTML = mat[i][j];
        }
    }
}

testBegin.onclick = () => {
    let upload = testFile.files[0];
    let reader = new FileReader();
    reader.readAsText(upload, 'UTF-8');
    reader.onload = testLoad;
    reader.onerror = testError;
}