## Trabalho final da disciplina de Robótica Móvel (CI1020) - Ciência da Computação - UFPR

*   Controle de um carro autônomo no ambiente simulado da Skoods.

### Ambiente da simulação

Acesse o sistema da [Skoods](https://github.com/skoods-org/Welcome) para baixar o ambiente da simulação.

Além disso vale a pena dar uma conferida no projeto open source, [Microsoft AirSim](https://github.com/microsoft/AirSim).

### Dataset

O formato padrão do *dataset* possui duas estruturas: um diretório com as imagens `(Ex.: images/)` e um arquivo de log com os *steering commands* do carro no formato *tsv - Tab Separated Values* `(Ex.: airsim_rec.txt)` .

Para ajudar na hora de gravar os dados, `airsim_dataset.py` é um script que possui uma classe que grava os dados da simulação, e gera um dataset novo em folha no padrão visto acima.

O script `__main__.py` já possui integração da classe de `airsim_dataset.py` . 

### Treinamento da rede neural

`train_nvidia_model.py` : Script de treinamento de uma rede neural convolucional baseada no modelo proposto pela NVIDIA no paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).

### Teste da rede neural

`test_nvidia_model.py` : Script de teste da rede neural convolucional em um ambiente de simulação AirSim. Serve principalmente para debugar a rede neural.

### Rodar a simulação

`__main__.py` : Script principal para rodar o carro autônomo tanto com o controle PID simples fornecido pela própria Skoods, quanto com a rede neural convolucional. **Lembre-se de sempre rodar o ambiente da simulação antes de rodar esse script**.
