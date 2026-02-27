import tf, { train } from '@tensorflow/tfjs-node';

async function traninModel(inputXs, outputYs) {
    // Criamos um modelo sequencial
    const model = tf.sequential();

    //primeira cada de rede:
    // entrada de 5 posições (alto, baixo, pequeno, medio, grande)

    //adicionar 150 neurons porque tem pouca base de treino
    //quanto mais neurons, mais capacidade de aprender padrões complexos e mais processamento 

    //ativação relu age como um filtro.
    // é como se ela deixasse somente os dados interessantes passarem, ou seja, os dados que tem um valor positivo.
    // se for 0 ou negatibo pode jogar fora, porque não tem valor para o modelo aprender.
    model.add(tf.layers.dense({inputShape: [5], units: 150, activation: 'relu'}));
    
    //saida com 3 neuroions porque tem 3 categorias (hatch, sedan, suv)
    //ativavation: softmax normaliza a saida em probabilidades, ou seja, a soma das saídas será igual a 1. Isso é útil para classificação, pois o modelo pode atribuir uma probabilidade a cada categoria.
    model.add(tf.layers.dense({units: 3, activation: 'softmax'})); // saída de 3 categorias

    // Compilamos o modelo com otimizador e função de perda
    // exemplo classico: classificação de imagems, recomendação e categoriazação 
    model.compile({
        optimizer: 'adam', // adaptive moment estimation, é um otimizador eficiente para muitos problemas de aprendizado de máquina aprende com historico de erros
        loss: 'categoricalCrossentropy', ///compara o que o modelo acha que é a resposta correta (as probabilidades) com a resposta correta (one-hot encoded) e calcula o erro. O modelo tenta minimizar esse erro durante o treinamento.
        metrics: ['accuracy'] //quanto mais distante de 1, mais errado o modelo está. O modelo tenta maximizar a acurácia durante o treinamento.
    });

    // Treinamos o modelo
     await model.fit(inputXs, outputYs, {
        epochs: 100,// passa 100 vezes por todo modelo
        shuffle: true,// embaralha os dados a cada época para evitar que o modelo aprenda padrões específicos da ordem dos dados, o que pode levar a um melhor desempenho geral.
        verbose: 0// desabilita o log interno do treinamento, para evitar poluição do console. Se quiser ver o progresso, pode usar verbose: 1 ou 2.
    });

    return model;
}

async function predict(model, pessoaNormalizada) {
    //trans formar o array em tensor
    const tfInput = tf.tensor2d(pessoaNormalizada); // transforma o array em tensor 2D (1 linha, 7 colunas)
    const prediction = model.predict(tfInput); // faz a previsão usando o modelo treinado
    const predArray = prediction.arraySync(); // converte a previsão para array
    return predArray[0].map((prop, index) => ({ prop, index })); // mapeia as probabilidades para as categorias correspondentes  
}

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorCarros = [
    [ 0, 1, 1, 0, 0], // fiesta
    [ 0, 1, 1, 0, 0], // hb20
    [ 0, 1, 0, 0, 1], // corolla
    [ 1, 0, 0, 1, 0],  // creta
    [ 1, 0, 1, 0, 0]  // tracker
]

// Labels das categorias a serem previstas (one-hot encoded)
const labelsNomes = ["hatch", "sedan", "suv"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // hatch - fiesta
    [1, 0, 0], // hatch - hb20
    [0, 1, 0], // sedan - corolla
    [0, 0, 1],  // suv - creta
    [0, 0, 1]
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorCarros)
const outputYs = tf.tensor2d(tensorLabels)
    
//quanto mais dado melhor 
// assim o algoritmo tem mais exemplos para aprender padrões e generalizar melhor. Com poucos dados, o modelo pode ter dificuldade em aprender e pode acabar decorando os exemplos de treino (overfitting) em vez de aprender a generalizar para novos dados.
const model = await traninModel(inputXs, outputYs);

//normalizando a idade de outra pessoa
const novaPessoa = {carro: "polo",altura:"abaixo",tamanhoPortaMalas:"pequeno"};

const novaCarro = [[0, 1, 1, 0, 0]]; // idade normalizada + one-hot encoding para cor e localização

//await predict(model, novaPessoaNormalizada)

const previsao = await predict(model, novaCarro);
const resultado = previsao.sort((a, b) => b.prop - a.prop)
                          .map(item => `${labelsNomes[item.index]} (${(item.prop * 100).toFixed(2)}%)`)
                          .join('\n'); // pega a categoria com maior probabilidade 

console.log(resultado)
