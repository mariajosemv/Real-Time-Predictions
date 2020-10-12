// testing this model
async function getData() {
    const datosCasasR = await fetch("datos.json");
    const datosCasas = await datosCasasR.json();
    var datosLimpios = datosCasas.map(casa => ({
        precio: casa.Precio,
        cuartos: casa.NumeroDeCuartosPromedio
    }))
    datosLimpios = datosLimpios.filter(casa => (casa.precio != null && casa.cuartos != null))
    return datosLimpios;
}
function visualizarDatos(data) {
    const valores = data.map(d => ({ x: d.cuartos, y: d.precio }));
    tfvis.render.scatterplot(
        { name: "Cuartos vs Precio" },
        { values: valores },
        {
            xLabel: "Cuartos",
            yLabel: "Precios",
            height: 300
        }
    );
}

function crearModelo() {
    const modelo = tf.sequential();

    modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    modelo.add(tf.layers.dense({ units: 1, useBias: true }));
    return modelo;
}

const optimizador = tf.train.adam();
const funcion_perdida = tf.losses.meanSquaredError;
const metricas = ['mse'];

async function entrenarModelo(model, inputs, labels) {
    model.compile({
        optimizer: optimizador,
        loss: funcion_perdida,
        metrics: metricas
    });


    const surface = { name: 'show.history live', tab: 'Training' }
    const tamanioBatch = 28;
    const epochs = 50;
    const history = [];

    return await model.fit(inputs, labels, {
        tamanioBatch,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}


function convertirDatosATensores(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const entradas = data.map(d => d.cuartos);
        const etiquetas = data.map(d => d.precio);
        const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
        const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

        const entradasMax = tensorEntradas.max();
        const entradasMin = tensorEntradas.min();
        const etiquetasMax = tensorEtiquetas.max();
        const etiquetasMin = tensorEtiquetas.min();

        //(dato-min)/(max-min)
        const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
        const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(entradasMin));

        return {
            entradas: entradasNormalizadas,
            etiquetas: etiquetasNormalizadas,
            entradasMax,
            entradasMin,
            etiquetasMax,
            etiquetasMin
        }
    })
}

var modelo;

async function run() {
    const data = await getData();

    visualizarDatos(data);
    modelo = crearModelo()

    const tensorData = convertirDatosATensores(data);
    const { entradas, etiquetas } = tensorData;
    await entrenarModelo(modelo, entradas, etiquetas)
}

run();
