var stopTraining;
const optimizador = tf.train.adam()
const funcion_perdida = tf.losses.meanSquaredError;
const metricas = ['mse'];
const metrics = ['loss', 'val_loss', 'mse', 'val_mse']


async function getData() {
    const datosCasasR = await fetch('https://raw.githubusercontent.com/mariajosemv/Real-Time-Predictions/master/datos.json');
    const datosCasas = await datosCasasR.json();
    const datosLimpios = datosCasas.map(casa => ({
      precio: casa.Precio,
      cuartos: casa.NumeroDeCuartosPromedio
    }))
    .filter(casa => (casa.precio != null && casa.cuartos != null));

    return datosLimpios;
  }

  function visualizarDatos(data){

    const visorInstance = tfvis.visor();
    if (!visorInstance.isOpen()) {
    visorInstance.toggle();
}

    const valores = data.map(d => ({
      x: d.cuartos,
      y: d.precio,
    }));

    tfvis.render.scatterplot(
      {name: 'Room vs Price'},
      {values: valores},
      {
        xLabel: 'Rooms',
        yLabel: 'Price',
        height: 300,
        
      }
    );
  }

function convertirDatosATensores(data){
  return tf.tidy(() => {
    tf.util.shuffle(data);
  
    const entradas = data.map(d => d.cuartos)
    const etiquetas = data.map(d => d.precio);
  
    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
    const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);
  
  
    const entradasMax = tensorEntradas.max();
    const entradasMin = tensorEntradas.min();
    const etiquetasMax = tensorEtiquetas.max();
    const etiquetasMin = tensorEtiquetas.min();
  
    // (dato -min) / (max-min)
    const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
    const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));
  
      return {
        entradas: entradasNormalizadas,
        etiquetas: etiquetasNormalizadas,
        // Return the min/max bounds so we can use them later.
        entradasMax,
        entradasMin,
        etiquetasMax,
        etiquetasMin,
      }
  
    });
  }

function crearModelo(){
  const modelo = tf.sequential();

  // agregar capa oculta que va a recibir 1 dato
  modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // agregar una capa de salida que va a tener 1 sola unidad
  modelo.add(tf.layers.dense({ units: 1, useBias: true }));

  return modelo;
}

async function entrenarModelo(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: optimizador,
    loss: funcion_perdida,
    metrics: metricas,
  });

  const surface = { name: 'show.history live', tab: 'Training' };
  const tamanioBatch = 28;
  const epochs = 50;
  const history = [];

  const trainDataSize = 3500;
  const testDataSize = 1500;

  const [xTrain, xTest] = tf.split(inputs, [trainDataSize, testDataSize]);
  const [yTrain, yTest] = tf.split(labels, [trainDataSize, testDataSize]);
  
  //const fitCallbacks = tfvis.show.fitCallbacks(surface, metrics);

  return await model.fit(xTrain, yTrain, {
    tamanioBatch,
    epochs,
    validationData: [xTest, yTest],
    shuffle: true,
    callbacks: {
      
      onEpochEnd: (epoch, log) => {
        history.push(log);
        tfvis.show.history(surface, history,  ['loss', 'mse', 'val_loss', 'val_mse']);

        if(stopTraining){
          modelo.stopTraining = true;
        }
      }
    }
  });
}

async function verCurvaInferencia(){
  var data = await getData();
  var tensorData = await convertirDatosATensores(data);

  const { entradasMax, entradasMin, etiquetasMin, etiquetasMax} = tensorData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0,1,100);
    const preds = modelo.predict(xs.reshape([100,1]));

    // (dato -min) / (max-min)  --> normalizar
    // dato = (max - min) + min --> desnormalizar

    const desnormX = xs
      .mul(entradasMax.sub(entradasMin))
      .add(entradasMin)

      const desnormY = preds
      .mul(etiquetasMax.sub(etiquetasMin))
      .add(etiquetasMin)

      return [desnormX.dataSync(), desnormY.dataSync()];

  });

  const puntosPrediccion = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const puntosOriginales = data.map(d => ({
    x: d.cuartos, y: d.precio,
  }));


  tfvis.render.scatterplot(
    { name: 'Prediction vs Original values', 
      tab:  'Predictions'},
    { values: [puntosOriginales, puntosPrediccion], series: ['original values','predictions']},
    {
      xLabel: 'Rooms',
      yLabel: 'Price',
      height: 300
    }
  );

}

async function cargarModelo(){
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');

  modelo = await tf.loadLayersModel(tf.io.browserFiles(
    [uploadJSONInput.files[0], uploadWeightsInput.files[0]]
  ));
  console.log("Modelo Cargado");
}

async function guardarModelo(){
    const saveResult = await modelo.save('downloads://modelo-regresion');

}


// Workflow

async function showData() {
    const data = await getData();
    visualizarDatos(data);

}

async function trainModel(){
    const data = await getData();
    modelo = crearModelo();
    const tensorData = convertirDatosATensores(data);
    const {entradas, etiquetas} = tensorData;

    await entrenarModelo(modelo, entradas, etiquetas);

}