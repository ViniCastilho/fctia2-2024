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

function trainingLoad(event) {
    let rows = event.target.result.split('\r\n');
    if (rows.length <= 1) rows = event.target.result.split('\n');
}

function trainingError(event) {
    alert('Um erro ocorreu ao ler o arquivo de treinamento.');
}

trainLoad.onclick = () => {
    let upload = trainFile.files[0];
    let reader = new FileReader();
    reader.readAsText(upload, 'UTF-8');
    reader.onload = trainingLoad;
    reader.onerror = trainingError;
}