function logistic(x) {
    return 1/(1+Math.exp(-x));
}

function logisticDerived (x) {
    return logistic(x) * (1 - logistic(x));
}

function hyperbolic(x) {
    return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x));
}

function hyperbolicDerived(x) {
    return 1 - (hyperbolic(x) ** 2);
}

const training = [
    [1,19,35,28,17,4,1],
    [4,22,38,32,14,9,1],
    [0,18,36,26,20,9,1],
    [84,37,20,72,100,47,2],
    [81,41,21,73,101,50,2],
    [82,43,22,68,101,48,2],
    [27,59,87,28,16,19,3],
    [17,69,74,29,19,21,3],
    [10,71,78,33,20,21,3],
    [94,15,26,78,64,37,4],
    [93,23,33,70,63,32,4],
    [92,23,33,65,58,45,4],
    [20,73,59,83,82,69,5],
    [15,74,60,83,86,73,5],
    [18,73,66,81,83,74,5],
];

const MAX_ERROR = 0.2;

const INPUT_NEURONS = 6;
const HIDDEN_NEURONS = 3;
const OUTPUT_NEURONS = 5;

const NETWORK_LAYER_SIZE = [INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS];
const LINK_COUNT = NETWORK_LAYER_SIZE.length-1;

let weightLayers = [];
let weightedSumLayers = [];
let errorLayers = [];

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

// Propagação direta (Forward Propagation)
function forwardPropagation(input) {
    let activations = [input]; // Ativações das camadas
    for (let layer = 0; layer < LINK_COUNT; layer++) {
        let layerActivation = [];
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            let weightedSum = 0;
            for (let p = 0; p < NETWORK_LAYER_SIZE[layer]; p++) {
                weightedSum += weightLayers[layer][n][p] * activations[layer][p];
            }
            weightedSumLayers[layer][n] = weightedSum; // Salvar somatório
            // Aplicar função de ativação
            layerActivation.push(logistic(weightedSum));
        }
        activations.push(layerActivation);
    }
    return activations;
}

// Retropropagação (Backpropagation)
function backPropagation(activations, expectedOutput) {
    // Calcular erro na camada de saída
    let outputLayerIndex = LINK_COUNT - 1;
    for (let n = 0; n < NETWORK_LAYER_SIZE[outputLayerIndex + 1]; n++) {
        let output = activations[outputLayerIndex + 1][n];
        let expected = (n + 1 === expectedOutput) ? 1 : 0; // Classe esperada
        errorLayers[outputLayerIndex][n] = (output - expected) * logisticDerived(weightedSumLayers[outputLayerIndex][n]);
    }

    // Propagar erro para as camadas ocultas
    for (let layer = outputLayerIndex - 1; layer >= 0; layer--) {
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            let errorSum = 0;
            for (let nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZE[layer + 2]; nextNeuron++) {
                errorSum += weightLayers[layer + 1][nextNeuron][n] * errorLayers[layer + 1][nextNeuron];
            }
            errorLayers[layer][n] = errorSum * logisticDerived(weightedSumLayers[layer][n]);
        }
    }
}

// Atualização dos pesos
function updateWeights(activations, learningRate) {
    for (let layer = 0; layer < LINK_COUNT; layer++) {
        for (let n = 0; n < NETWORK_LAYER_SIZE[layer + 1]; n++) {
            for (let p = 0; p < NETWORK_LAYER_SIZE[layer]; p++) {
                weightLayers[layer][n][p] -= learningRate * errorLayers[layer][n] * activations[layer][p];
            }
        }
    }
}

// Treinamento da rede
function trainNetwork(epochs, learningRate) {
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0;
        for (let data of training) {
            let input = data.slice(0, INPUT_NEURONS); // Entradas
            let expectedOutput = data[INPUT_NEURONS]; // Classe esperada
            let activations = forwardPropagation(input); // Propagação direta
            backPropagation(activations, expectedOutput); // Retropropagação
            updateWeights(activations, learningRate); // Atualização dos pesos

            // Calcular erro total
            let outputLayer = activations[LINK_COUNT];
            for (let n = 0; n < OUTPUT_NEURONS; n++) {
                let expected = (n + 1 === expectedOutput) ? 1 : 0;
                totalError += 0.5 * Math.pow(outputLayer[n] - expected, 2);
            }
        }
        console.log(`Epoch ${epoch + 1}, Total Error: ${totalError}`);
        if (totalError < MAX_ERROR) break;
    }
}

// Executar treinamento
trainNetwork(1000, 0.1); // 1000 épocas, taxa de aprendizado 0.1
